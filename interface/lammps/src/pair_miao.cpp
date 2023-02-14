#include <mpi.h>
#include <stdlib.h>
#include <pair_miao.h>
#include <string>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "utils.h"


using namespace LAMMPS_NS;
using namespace std;


PairMIAO::PairMIAO(LAMMPS *lmp) : Pair(lmp) 
{
}


PairMIAO::~PairMIAO() 
{
    if (allocated) 
    {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }
    // delete the map from the global index to local index
    atom->map_delete();
    atom->map_style = Atom::MAP_NONE;
}


void PairMIAO::allocate()
{
    int n = atom->ntypes;
    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            setflag[i][j] = 1;
    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
    allocated = 1;
}

void PairMIAO::coeff(int narg, char** arg)
{
    if (!allocated)
        allocate();
}

void PairMIAO::settings(int narg, char **arg)
{
    if (narg != 3)
        error->all(FLERR, "Illegal pair_style command");
    model_filename = arg[0];
    #if LAMMPS_VERSION_NUMBER >= 20200918
        cutoff = utils::numeric(FLERR, arg[1], false, lmp);
    #else
        cutoff = force->numeric(FLERR, arg[1]);
    #endif
    cutoffsq = cutoff * cutoff;
    model_prec = arg[2];
}


void PairMIAO::init_style()
{
    // use the new neighbor list starting from this version
    #if LAMMPS_VERSION_NUMBER >= 20220324
        neighbor->add_request(this, NeighConst::REQ_FULL);
    #else
        int irequest = neighbor->request(this, instance_me);
        neighbor->requests[irequest]->pair = 1;
        neighbor->requests[irequest]->half = 0;
        neighbor->requests[irequest]->full = 1;
    #endif
    int n = atom->ntypes;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            cutsq[i][j] = cutoffsq;
    try
    {
        //enable the optimize of torch.script
        torch::jit::GraphOptimizerEnabledGuard guard{true};
        torch::jit::setGraphExecutorOptimize(true);
        // load the model
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_filename);
        if (model_prec == "float") tensor_type = torch::kFloat32;
        // freeze the module
        int id;
        if (torch::cuda::is_available()) 
        {
            device = torch::kCUDA;
            cout << "The simulations are performed on the GPU" << endl;
            option1 = option1.pinned_memory(true);
            option2 = option2.pinned_memory(true);
            module.to(device);
        }
        module.eval();
        module = torch::jit::optimize_for_inference(module);
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "error loading the model\n";
    }
    std::cout << "ok\n";

    // create the map from global to local
    if (atom->map_style == Atom::MAP_NONE) {
        atom->nghost = 0;
        atom->map_init(1);
        atom->map_set();
    }
}


double PairMIAO::init_one(int i, int j) { return cutoff; }


void PairMIAO::set_properties()
{
    properties = {"forces"};
    if (eflag_global) properties.push_back("energy");
    if (eflag_atom) properties.push_back("site_energy");
    if (vflag_atom) error->all(FLERR, "Not support per-atom virial, please check vflag.");
    if (vflag_global) properties.push_back("virial");
}


void PairMIAO::compute(int eflag, int vflag)
{
    #if LAMMPS_VERSION_NUMBER >= 20190329
        ev_init(eflag, vflag);
    #else
        if (eflag || vflag) ev_setup(eflag, vflag); 
        else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;
    #endif
    set_properties();
    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type; 
    tagint *tag = atom->tag;
    int nlocal = atom->nlocal, nghost = atom->nghost;
    int *ilist, *jlist, *numneigh, **firstneigh;
    int i, ii, inum, j, jj, jnum, maxneigh;
    int totneigh = 0, nall = nghost + nlocal;

    torch::Dict<std::string, torch::Tensor> batch_data;
    torch::Dict<std::string, torch::Tensor> results;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    int numneigh_atom = accumulate(numneigh, numneigh + inum , 0);

    std::vector<double> cart(nall * 3);
    std::vector<long> atom_index(numneigh_atom * 2);
    std::vector<long> ghost_neigh(numneigh_atom);
    std::vector<long> local_species(inum);
    std::vector<long> batch(nall);
    double dx, dy, dz, d2;
    // assign the cart with x
    for (ii = 0; ii < nall; ++ii)
    {
        cart[3 * ii] = x[ii][0];
        cart[3 * ii + 1] = x[ii][1];
        cart[3 * ii + 2] = x[ii][2];
        batch[ii] = 0;
    }

    for (ii = 0; ii < inum; ++ii)
    {
        i = ilist[ii];
        local_species[i] = type[i] - 1;
        for (jj = 0; jj < numneigh[i]; ++jj)
        {
            j = firstneigh[i][jj];
            dx = x[j][0] - x[i][0];
            dy = x[j][1] - x[i][1];
            dz = x[j][2] - x[i][2];
            d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < cutoffsq)
            {
                atom_index[totneigh * 2] = i;
                atom_index[totneigh * 2 + 1] = atom->map(tag[j]);
                ghost_neigh[totneigh] = j;
                ++totneigh;
            }
        }
    }
    
    // test
    /*
    ofstream cart_file("cart.txt", ios::app);
    for (ii = 0; ii < nall * 3; ++ii){
        cart_file << cart[ii] << " ";
    }
    cart_file.close();

    ofstream batch_file("batch.txt", ios::app);
    for (ii = 0; ii < inum; ++ii){
        batch_file << batch[ii] << " ";
    }
    batch_file.close();

    ofstream atom_file("atom_index.txt", ios::app);
    for (ii = 0; ii < totneigh * 2; ++ii){
        atom_file << atom_index[ii] << " ";
    }
    atom_file.close();

    ofstream ghost_file("ghost.txt", ios::app);
    for (ii = 0; ii < totneigh; ++ii){
        ghost_file << ghost_neigh[ii] << " ";
    }
    ghost_file.close();

    ofstream species_file("species.txt", ios::app);
    for (ii = 0; ii < inum; ++ii){
        species_file << local_species[ii] << " ";
    }
    species_file.close();
    */
    auto cart_ = torch::from_blob(cart.data(), {nall, 3}, option1).to(device, true).to(tensor_type);
    auto atom_index_ = torch::from_blob(atom_index.data(), {totneigh, 2}, option2).transpose(1, 0).to(device, true);
    auto ghost_neigh_ = torch::from_blob(ghost_neigh.data(), {totneigh}, option2).to(device, true);
    auto local_species_ = torch::from_blob(local_species.data(), {inum}, option2).to(device, true);
    auto batch_ = torch::from_blob(batch.data(), {inum}, option2).to(device, true);
    batch_data.insert("coordinate", cart_);
    batch_data.insert("atomic_number", local_species_);
    batch_data.insert("edge_index", atom_index_);
    batch_data.insert("batch", batch_);
    batch_data.insert("ghost_neigh", ghost_neigh_);

    torch::IValue output = module.forward({batch_data, properties});
    results = c10::impl::toTypedDict<std::string, torch::Tensor>(output.toGenericDict());

    auto forces_tensor = results.at("forces_p").to(torch::kDouble).cpu().reshape({-1});
    auto forces = forces_tensor.data_ptr<double>();
    // forces
    for (i = 0; i < nall; ++i)
        for (j = 0; j < 3; ++j)
            f[i][j] = forces[i*3+j];

    if (eflag_global)
    {
        auto energy_tensor = results.at("energy_p").to(torch::kDouble).cpu();
        auto energy = energy_tensor.data_ptr<double>();
        ev_tally(0, 0, nlocal, 1, energy[0], 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    if (eflag_atom)
    {
        auto site_energy_tensor = results.at("site_energy_p").to(torch::kDouble).cpu();
        auto site_energy = site_energy_tensor.data_ptr<double>();
        for (ii = 0; ii < nlocal; ++ii)
        {
            i = ilist[ii];
            eatom[ii] = site_energy[i];
        }
    }
    if (vflag_global)
    {
        auto virial_tensor = results.at("virial_p").to(torch::kDouble).cpu().reshape({-1});
        auto virial_data = virial_tensor.data_ptr<double>();
        virial[0] = virial_data[0];
        virial[1] = virial_data[4];
        virial[2] = virial_data[8];
        virial[3] = (virial_data[1] + virial_data[3]) / 2;
        virial[4] = (virial_data[2] + virial_data[6]) / 2;
        virial[5] = (virial_data[5] + virial_data[7]) / 2;
    }

    if (vflag_fdotr) virial_fdotr_compute();
}
