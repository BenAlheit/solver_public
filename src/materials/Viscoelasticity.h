//
// Created by alhei on 2022/09/14.
//

#ifndef SOLVER_VISCOELASTICITY_H
#define SOLVER_VISCOELASTICITY_H

#include "Material.h"
#include "MaterialState.h"

// TODO Include calculation of energetic dissipation due to viscous behaviour. Store this value in 'state'
template<unsigned int dim>
class Viscoelasticity : public Material<dim> {
public:
    Viscoelasticity(ElasticMaterial<dim> *e_law,
                    const vector<double> &beta,
                    const vector<double> &tau)
            : Material<dim>(e_law->get_id()), elastic_law(e_law), beta(beta), tau(tau), n(beta.size()) {};

    void update_stress(const double &dt) override {
        throw NotImplemented("'update_stress' has not been implemented for 'Viscoelasticity'. "
                             "Use 'update_stress_and_tangent' instead.");
    };

    void update_tangent(const double &dt) override {
        throw NotImplemented("'update_tangent' has not been implemented for 'Viscoelasticity'. "
                             "Use 'update_stress_and_tangent' instead.");
    };

    void update_stress_and_tangent(const double &dt) override;

    typedef ViscoelasticState<dim> state_type;

    void set_state(StateBase<dim> *new_ptr) override { this->state = dynamic_cast<state_type *>(new_ptr); };

    StateBase<dim> *create_state() const override { return dynamic_cast<StateBase<dim> *>(new state_type(n)); }

private:
    state_type *state;
    array<unsigned int, dim> range;
    ElasticMaterial<dim> *elastic_law;
    const vector<double> beta;
    const vector<double> tau;
    const unsigned int n;
};

template<unsigned int dim>
void Viscoelasticity<dim>::update_stress_and_tangent(const double &dt) {
    Tensor<2, dim> G, G_T, tau_inf_n, tau_inf_n1;
    G = invert(state->F_n) * state->F_n1;
    G_T = transpose(G);
    tau_inf_n = this->state->tau_inf_n;

    elastic_law->set_state(state);
    elastic_law->update_stress_and_tangent(dt);
    SymmetricTensor<4, dim> c_inf_n1 = this->state->c_n1;
    tau_inf_n1 = this->state->tau_n1;

    double xi;
    double beta_xi = 0;

    for (unsigned int alpha = 0; alpha < n; alpha++) {
        xi = exp(-dt / (2. * tau.at(alpha)));
        beta_xi += beta.at(alpha) * xi;
        state->tau_v_n1.at(alpha) = xi * (beta.at(alpha) * tau_inf_n1
                                           + G_T * (xi * state->tau_v_n.at(alpha) - beta.at(alpha) * tau_inf_n) * G);
        this->state->tau_n1 += state->tau_v_n1.at(alpha);
    }

    this->state->c_n1 *= (1 + beta_xi);
    this->state->tau_inf_n1 = tau_inf_n1;
}

#endif //SOLVER_VISCOELASTICITY_H
