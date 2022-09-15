//
// Created by alhei on 2022/09/07.
//


#ifndef SOLVER_RATEDEPENDENTPLASTICITY_H
#define SOLVER_RATEDEPENDENTPLASTICITY_H

#include "MaterialState.h"
#include "Material.h"
#include "RateIndependtPlasticity.h"
#include <deal.II/base/tensor.h>

using namespace std;
using namespace dealii;

template<unsigned int dim>
class RateDependence {
public:

    RateDependence(PlasticityTheory<dim> *p_theory) : p_theory(p_theory) {};

    virtual double flow_rate(const Tensor<2, dim> &tau, const double &ep) = 0;

protected:
    PlasticityTheory<dim> *p_theory;
};

template<unsigned int dim>
class PowerLaw : public RateDependence<dim> {
public:
    PowerLaw(const double &mu, const double &m, PlasticityTheory<dim> *p_theory)
            : RateDependence<dim>(p_theory), mu(mu), m(m) {};

protected:
    double mu;
    double m;
};

template<unsigned int dim>
class Peric : public PowerLaw<dim> {
public:
    Peric(const double &mu, const double &m, PlasticityTheory<dim> *p_theory)
            : PowerLaw<dim>(mu, m, p_theory) {};

    double flow_rate(const Tensor<2, dim> &tau, const double &ep) override {
        const double f = this->p_theory->yield_surface->f(tau, ep);
        if (f < 0) { return 0; }
        else {
            double tbar_norm = sqrt(3. / 2.) *(tau - Physics::Elasticity::StandardTensors<dim>::I * trace(tau) / 3.).norm();
            return (pow(tbar_norm/this->p_theory->yield_surface->hardening_law->sig_y(ep), this->m) - 1.) / this->mu;
        }
    };
};

//TODO complete
template<unsigned int dim>
class RateDependentPlasticity : public Material<dim> {
public:
    RateDependentPlasticity(RateDependence<dim> * rate_dependent_theory)
    : rate_dependent_theory(rate_dependent_theory) {
        throw NotImplemented("'RateDependentPlasticity' has not been sufficiently implemented yet.");
    };

    void update_stress(const double &dt) override;

    void update_tangent(const double &dt) override {
        throw NotImplemented("'update_stress' has not been implemented for 'RateDependentPlasticity'. "
                             "Use 'update_stress_and_tangent' instead.");
    };

    void update_stress_and_tangent(const double &dt) override{
        update_stress(dt);
        Material<dim>::approximate_tangent(dt, state);
    };

    typedef RateDependentPlasticityState<dim> state_type;

    void set_state(StateBase<dim> *new_ptr) override { this->state = dynamic_cast<state_type *>(new_ptr); };

    StateBase<dim> *create_state() const override { return dynamic_cast<StateBase<dim> *>(new state_type()); }

private:
    RateDependence<dim> * rate_dependent_theory;
    state_type *state;
};

template<unsigned int dim>
void RateDependentPlasticity<dim>::update_stress(const double &dt) {

}

#endif //SOLVER_RATEDEPENDENTPLASTICITY_H
