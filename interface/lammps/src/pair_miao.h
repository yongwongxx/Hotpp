#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS

PairStyle(miao, PairMIAO)

#else

#ifndef LMP_PAIR_MIAO_H
#define LMP_PAIR_MIAO_H

#include "pair.h"
#include <torch/torch.h>
#include <torch/script.h> 
#include <string>

namespace LAMMPS_NS 
{
    class PairMIAO : public Pair 
    { 
        public:
            torch::jit::script::Module module;
            PairMIAO(class LAMMPS *);
            virtual ~PairMIAO();
            virtual void compute(int, int);
            virtual void init_style();
            virtual double init_one(int, int);
            virtual void settings(int, char **);
            virtual void coeff(int, char **);
        protected:
            std::vector<std::string> properties{"forces"};
            virtual void allocate();
            virtual void set_properties();
            double cutoff; 
            double cutoffsq;
            std::string model_filename;
            std::string model_prec;
            torch::Dtype tensor_type=torch::kDouble;
            torch::TensorOptions option1=torch::TensorOptions().dtype(torch::kDouble);
            torch::TensorOptions option2=torch::TensorOptions().dtype(torch::kLong);
            torch::DeviceType device=torch::kCPU;
    };
}

#endif
#endif
