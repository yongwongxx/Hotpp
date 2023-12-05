/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(miao/dipole, ComputeMiaoDipole);
// clang-format on
#else

#ifndef LMP_COMPUTE_MIAO_DIPOLE_H
#define LMP_COMPUTE_MIAO_DIPOLE_H

#include "compute.h"
#include <torch/torch.h>
#include <torch/script.h> 
#include <string>

namespace LAMMPS_NS 
{
    class ComputeMiaoDipole : public Compute 
    {
        public:
            torch::jit::script::Module module;
            ComputeMiaoDipole(class LAMMPS *, int, char **);
            ~ComputeMiaoDipole();
            void init();
            void init_list(int, class NeighList *);
            void compute_vector();
        
        private:
            std::vector<std::string> properties{"dipole"};
            double cutoff; 
            double cutoffsq;
            std::string model_filename;
            std::string model_prec;
            torch::Dtype tensor_type=torch::kDouble;
            torch::TensorOptions option1=torch::TensorOptions().dtype(torch::kDouble);
            torch::TensorOptions option2=torch::TensorOptions().dtype(torch::kLong);
            torch::DeviceType device=torch::kCPU;
            class NeighList *list;

            int compute_pairs(int);
            void reallocate(int);
    };

}    // namespace LAMMPS_NS

#endif
#endif
