#ifndef SOLVER_EXPLICITRATEDEPENDENTCRYSTALPLASTICITYCLICALTENSIONCOMPRESSION_H
#define SOLVER_EXPLICITRATEDEPENDENTCRYSTALPLASTICITYCLICALTENSIONCOMPRESSION_H

#include "CyclicalTensionCompression.h"
#include "../../materials/Viscoelasticity.h"
#include "../../materials/ExplicitRateDependentPlasticity.h"
#include "../../materials/ExplicitRateDependentCrystalPlasticity.h"

template<unsigned int dim>
class ExplicitRateDependentCrystalPlasticityCyclicalTensionCompression {
public:
    explicit ExplicitRateDependentCrystalPlasticityCyclicalTensionCompression();

    map<unsigned int, Material<dim> *> materials;

private:

};

template<unsigned int dim>
ExplicitRateDependentCrystalPlasticityCyclicalTensionCompression<dim>::ExplicitRateDependentCrystalPlasticityCyclicalTensionCompression() {
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    auto elastic_base = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    const unsigned int n_sys = 1;
    const double visc = 1000;
    const double m = 0.1;
    const double sig_y = 60;
    const double q = 0.5;
    const double rot_deg = 90;

    const string name = "visco-cp-1-sys-ctc-"+ to_string((int)rot_deg)+"-deg";

    vector<double> ref_euler_angles(dim, 0);
    ref_euler_angles.at(2) = rot_deg;

    auto example_state = new SingleSlipCrystalPlasticityState<dim>(ref_euler_angles);

    materials[/*material id*/ 0] = new ExplicitRateCrystalDependentPlasticity<dim>(elastic_base,
                                                                                   example_state,
                                                                                   sig_y,
                                                                                   visc,
                                                                                   m,
                                                                                   q);

    const double base_dt = 10;
    const unsigned int n_cycles = 4;

    const double dt_ratio = 0.4;

    const double extension = 0.006;
    const double compression = 0.006;
    const unsigned int n_steps = 50;
    const unsigned int n_out = 25;

    vector<double> dts;
    dts.push_back(base_dt / 2);
    double current_dt = base_dt;
    for (unsigned int i = 0; i < n_cycles; i++) {
        dts.push_back(current_dt);
        current_dt *= dt_ratio;
        dts.push_back(current_dt);
    }


    vector<unsigned int> material_ids({0});
    map<nScalarOutput, vector<unsigned int>, nOutputHash> n_scalar_outputs;
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::H, n_sys), material_ids});
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::ta, n_sys), material_ids});
    n_scalar_outputs.insert({OutputFlags::nScalarOutput(OutputFlags::nScalarOutputFlag::nu, n_sys), material_ids});


    map<nVectorOutput, vector<unsigned int>, nOutputHash> n_vector_outputs;
    n_vector_outputs.insert({OutputFlags::nVectorOutput(OutputFlags::nVectorOutputFlag::M, n_sys), material_ids});
    n_vector_outputs.insert({OutputFlags::nVectorOutput(OutputFlags::nVectorOutputFlag::S, n_sys), material_ids});

    map<nTensorOutput, vector<unsigned int>, nOutputHash> n_tensor_outputs;
    n_tensor_outputs.insert({OutputFlags::nTensorOutput(OutputFlags::nTensorOutputFlag::SYS, n_sys), material_ids});

    CyclicalTensionCompression<dim> solve = CyclicalTensionCompression<dim>(name,
                                                                            materials,
                                                                            extension,
                                                                            compression,
                                                                            n_steps,
                                                                            n_out,
                                                                            dts,
                                                                            n_scalar_outputs,
                                                                            n_vector_outputs,
                                                                            n_tensor_outputs);
}


#endif //SOLVER_EXPLICITRATEDEPENDENTCRYSTALPLASTICITYCLICALTENSIONCOMPRESSION_H
