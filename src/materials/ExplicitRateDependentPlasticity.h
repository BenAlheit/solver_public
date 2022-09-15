#ifndef SOLVER_EXPLICITRATEDEPENDENTPLASTICITY_H
#define SOLVER_EXPLICITRATEDEPENDENTPLASTICITY_H

#include "Material.h"
#include "MaterialState.h"

template<unsigned int dim>
class ExplicitRateDependentPlasticity : public Material<dim> {
public:
    ExplicitRateDependentPlasticity(ElasticMaterial<dim> *e_law,
                                    const double &sig_y,
                                    const double &mu,
                                    const double &m)
            : Material<dim>(e_law->get_id()), elastic_law(e_law), sig_y(sig_y), mu(mu), m(m) {};

    void update_stress(const double &dt) override;

    void update_tangent(const double &dt) {
        throw NotImplemented("'update_tangent' has not been implemented for 'ExplicitRateDependentPlasticity'. "
                             "Use 'update_stress_and_tangent' instead.");
    };

    void update_stress_and_tangent(const double &dt) override;

    typedef ExplicitRateDependentPlasticityState<dim> state_type;

    void set_state(StateBase<dim> *new_ptr) override { this->state = dynamic_cast<state_type *>(new_ptr); };

    StateBase<dim> *create_state() const override { return dynamic_cast<StateBase<dim> *>(new state_type()); }

private:
    const double sig_y, mu, m;
    state_type *state;
    array<unsigned int, dim> range;
    ElasticMaterial<dim> *elastic_law;

};

template<unsigned int dim>
void ExplicitRateDependentPlasticity<dim>::update_stress(const double &dt) {
    if(almost_equals(state->ep_n1, state->ep_n)) {
        Tensor<2, dim> Tn, tau_bar_n, tau_n, Lp;
        double sig_bar, dln;
        tau_n = state->tau_n;
        tau_bar_n = tau_n - Physics::Elasticity::StandardTensors<dim>::I * trace(tau_n) / 3.;
        sig_bar = tau_bar_n.norm() * sqrt(3. / 2.);
        if (sig_bar > 0) {

            Tn = tau_bar_n / tau_bar_n.norm();
            dln = (1. / mu) * pow(sig_bar / sig_y, 1. / m);
            Lp = dln * Tn;

            state->Fp_n1 = state->Fp_n + dt * Lp * state->Fp_n;
            state->ep_n1 = state->ep_n + dt * fabs(dln);
        }
    }

    state->update_elastic_component();

    elastic_law->set_state(state->elastic_component);
    elastic_law->update_stress(dt);

    state->tau_n1 = state->elastic_component->tau_n1;
}

template<unsigned int dim>
void ExplicitRateDependentPlasticity<dim>::update_stress_and_tangent(const double &dt) {
    if(almost_equals(state->ep_n1, state->ep_n)) {
        Tensor<2, dim> Tn, tau_bar_n, tau_n, Lp;
        double sig_bar, dln;
        tau_n = state->tau_n;
        tau_bar_n = tau_n - Physics::Elasticity::StandardTensors<dim>::I * trace(tau_n) / 3.;
        sig_bar = tau_bar_n.norm() * sqrt(3. / 2.);
        if (sig_bar > 0) {

            Tn = tau_bar_n / tau_bar_n.norm();
            dln = (1. / mu) * pow(sig_bar / sig_y, 1. / m);
            Lp = dln * Tn;

            state->Fp_n1 = state->Fp_n + dt * Lp * state->Fp_n;
            state->ep_n1 = state->ep_n + dt * fabs(dln);
        }
    }

    state->update_elastic_component();

    elastic_law->set_state(state->elastic_component);
    elastic_law->update_stress_and_tangent(dt);

    state->tau_n1 = state->elastic_component->tau_n1;
    state->c_n1 = state->elastic_component->c_n1;
}

//template<unsigned int dim>
//void ExplicitRateDependentPlasticity<dim>::update_stress_and_tangent(const double &dt) {
//    update_stress(dt);
//    this->template approximate_tangent(dt, state, 1e-1);
////    this->template approximate_tangent(dt, state, 1e-1); // <- supprisingly large step required to calculate tangent accurately (works for cyclical loading example).
//}


#endif //SOLVER_EXPLICITRATEDEPENDENTPLASTICITY_H
