#ifndef SOLVER_CHECKAPPROXIMATETANGENTVISCOELASTIC_H
#define SOLVER_CHECKAPPROXIMATETANGENTVISCOELASTIC_H

#include "../../utils/utils.h"

template<unsigned int dim>
class CheckApproximateTangentViscoelastic{
public:
    CheckApproximateTangentViscoelastic();
};

template<unsigned int dim>
CheckApproximateTangentViscoelastic<dim>::CheckApproximateTangentViscoelastic() {
    const double kappa = 1;
    const double mu = 2;

    auto elastic_base = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);
    auto approx_base = new NeoHookean<dim>(/*material id*/ 1, kappa, mu);
    const vector<double> beta({0.}), tau({0.1});

    auto analytical = new Viscoelasticity<dim>(elastic_base, beta, tau);
    auto approx = new Viscoelasticity<dim>(approx_base, beta, tau);

    Tensor<2, dim> F_n = Physics::Elasticity::StandardTensors<dim>::I;
    Tensor<2, dim> F_n1 = Physics::Elasticity::StandardTensors<dim>::I;

//  Arbitrarily chosen
    F_n1[0][0] = 1.2;
    F_n1[0][1] = 0.2;
    F_n1[2][1] = 1.2;

    auto an_state = analytical->create_state();
    an_state->F_n1 = F_n1;
    analytical->set_state(an_state);
    analytical->update_stress_and_tangent(0.1);
    cout << "Analytical: " << endl;
    print_symmetric_4th_order<dim>(an_state->c_n1);
    cout << endl;



    auto ap_state = approx->create_state();
    ap_state->F_n1 = F_n1;
    approx->set_state(ap_state);
    approx->template approximate_tangent(0.1, ap_state);
    cout << "Approximate: " << endl;
    print_symmetric_4th_order<dim>(ap_state->c_n1);
    cout << endl;
    SymmetricTensor<4, dim> diff = ap_state->c_n1 - an_state->c_n1;
    SymmetricTensor<4, dim> avg = (ap_state->c_n1 + an_state->c_n1)/2.;
    cout << "% error: " << 100 * diff.norm() / avg.norm() << endl;

}

#endif //SOLVER_CHECKAPPROXIMATETANGENTVISCOELASTIC_H
