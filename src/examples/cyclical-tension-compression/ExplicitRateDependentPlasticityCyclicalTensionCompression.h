#ifndef SOLVER_EXPLICITRATEDEPENDENTPLASTICITYCLICALTENSIONCOMPRESSION_H
#define SOLVER_EXPLICITRATEDEPENDENTPLASTICITYCLICALTENSIONCOMPRESSION_H

#include "CyclicalTensionCompression.h"
#include "../../materials/Viscoelasticity.h"
#include "../../materials/ExplicitRateDependentPlasticity.h"

template<unsigned int dim>
class ExplicitRateDependentPlasticityCyclicalTensionCompression{
public:
    explicit ExplicitRateDependentPlasticityCyclicalTensionCompression();
    map<unsigned int, Material<dim> *> materials;

private:

};

template<unsigned int dim>
ExplicitRateDependentPlasticityCyclicalTensionCompression<dim>::ExplicitRateDependentPlasticityCyclicalTensionCompression() {
    const string name = "explicit-rate-dependent-plasticity-cyclical-tension-compression";
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    auto elastic_base = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

//    const double visc = 100;
//    const double m = 0.1;
//    const double visc = 1000;
    const double visc = 1;
    const double m = 0.1;
    const double sig_y = 60;

    materials[/*material id*/ 0] = new ExplicitRateDependentPlasticity<dim>(elastic_base,
                                                                            sig_y,
                                                                            visc,
                                                                            m);

    const double base_dt = 10;
    const unsigned int n_cycles = 4;

    const double dt_ratio = 0.4;

    const double extension = 0.004;
    const double compression = 0.004;
    const unsigned int n_steps = 50;
    const unsigned int n_out = 25;

    vector<double> dts;
    dts.push_back(base_dt/2);
    double current_dt = base_dt;
    for(unsigned int i = 0; i < n_cycles; i++){
        dts.push_back(current_dt);
        current_dt *= dt_ratio;
        dts.push_back(current_dt);
    }

    CyclicalTensionCompression<dim> solve = CyclicalTensionCompression<dim>(name,
                                                                            materials,
                                                                            extension,
                                                                            compression,
                                                                            n_steps,
                                                                            n_out,
                                                                            dts);
}


#endif //SOLVER_EXPLICITRATEDEPENDENTPLASTICITYCLICALTENSIONCOMPRESSION_H
