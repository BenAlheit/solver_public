#ifndef SOLVER_ELASTICCYCLICALTENSIONCOMPRESSION_H
#define SOLVER_ELASTICCYCLICALTENSIONCOMPRESSION_H

#include "CyclicalTensionCompression.h"

template<unsigned int dim>
class ElasticCyclicalTensionCompression{
public:
    explicit ElasticCyclicalTensionCompression();
    map<unsigned int, Material<dim> *> materials;
private:

};

template<unsigned int dim>
ElasticCyclicalTensionCompression<dim>::ElasticCyclicalTensionCompression() {
    const string name = "elastic-cyclical-tension-compression";
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    materials[/*material id*/ 0] = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    const double base_dt = 1;
    const unsigned int n_cycles = 4;

    const double dt_ratio = 0.5;

    const double extension = 0.2;
    const double compression = 0.2;
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


#endif //SOLVER_ELASTICCYCLICALTENSIONCOMPRESSION_H
