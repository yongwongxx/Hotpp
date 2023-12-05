// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_miao_dipole.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <numeric>

using namespace LAMMPS_NS;
using namespace std;

/* ---------------------------------------------------------------------- */

// compute COMPUTE_ID GROUP miao/dipole model.pt cutoff prec
ComputeMiaoDipole::ComputeMiaoDipole(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
    if (narg != 6) 
        error->all(FLERR,"Illegal compute miao/dipole command");
    if (igroup) 
        error->all(FLERR,"Compute pressure must use group all");

    model_filename = arg[3];
    #if LAMMPS_VERSION_NUMBER >= 20200918
        cutoff = utils::numeric(FLERR, arg[4], false, lmp);
    #else
        cutoff = force->numeric(FLERR, arg[4]);
    #endif
    cutoffsq = cutoff * cutoff;
    model_prec = arg[5];

    scalar_flag = 0;
    vector_flag = 1;
    size_vector = 3;
    extvector = 1;

    properties = {"dipole"};

    vector = new double[size_vector];
}

/* ---------------------------------------------------------------------- */

ComputeMiaoDipole::~ComputeMiaoDipole()
{
    // delete the map from the global index to local index
    atom->map_delete();
    atom->map_style = Atom::MAP_NONE;
    delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeMiaoDipole::init()
{
    // https://docs.lammps.org/Developer_notes.html
    #if LAMMPS_VERSION_NUMBER >= 20220324
        neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
    #else
        int irequest = neighbor->request(this, instance_me);
        neighbor->requests[irequest]->pair = 0;
        neighbor->requests[irequest]->compute = 1;
        neighbor->requests[irequest]->half = 0;
        neighbor->requests[irequest]->full = 1;
        neighbor->requests[irequest]->occasional = 1;
    #endif
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
            std::cout << "The simulations are performed on the GPU" << std::endl;
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


void ComputeMiaoDipole::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}


void ComputeMiaoDipole::compute_vector()
{
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

    neighbor->build_one(list);

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    int numneigh_atom = accumulate(numneigh, numneigh + inum , 0);

    std::vector<double> cart(nall * 3);
    std::vector<long> idx_i(numneigh_atom);            // atom i
    std::vector<long> idx_j(numneigh_atom);            // atom j
    std::vector<long> ghost_neigh(numneigh_atom);      // ghost atom j for calculate distance
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
                idx_i[totneigh] = i;
                idx_j[totneigh] = atom->map(tag[j]);
                ghost_neigh[totneigh] = j;
                ++totneigh;
            }
        }
    }
    
    std::vector<float> n_atoms={10.0};
    
    auto cart_ = torch::from_blob(cart.data(), {nall, 3}, option1).to(device, true).to(tensor_type);
    auto idx_i_ = torch::from_blob(idx_i.data(), {totneigh}, option2).to(device, true);
    auto idx_j_ = torch::from_blob(idx_j.data(), {totneigh}, option2).to(device, true);
    auto ghost_neigh_ = torch::from_blob(ghost_neigh.data(), {totneigh}, option2).to(device, true);
    auto local_species_ = torch::from_blob(local_species.data(), {inum}, option2).to(device, true);
    auto batch_ = torch::from_blob(batch.data(), {inum}, option2).to(device, true);
    auto n_atoms_ = torch::from_blob(n_atoms.data(), {1}, option1).to(device, true).to(tensor_type);
    batch_data.insert("coordinate", cart_);
    batch_data.insert("atomic_number", local_species_);
    batch_data.insert("idx_i", idx_i_);
    batch_data.insert("idx_j", idx_j_);
    batch_data.insert("batch", batch_);
    batch_data.insert("ghost_neigh", ghost_neigh_);
    batch_data.insert("n_atoms", n_atoms_);
    torch::IValue output = module.forward({batch_data, properties, false});
    results = c10::impl::toTypedDict<std::string, torch::Tensor>(output.toGenericDict());
    auto dipole_tensor = results.at("dipole_p").to(torch::kDouble).cpu().reshape({-1});
    auto dipole = dipole_tensor.data_ptr<double>();
    for (i = 0; i < 3; ++i)
    {
        vector[i] = dipole[i];
    }
}
