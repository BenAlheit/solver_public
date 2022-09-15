#ifndef SOLVER_RATEINDEPENDTPLASTICITY_H
#define SOLVER_RATEINDEPENDTPLASTICITY_H

#include "MaterialState.h"
#include "Material.h"
#include "cmath"

#include "../numerics/TimeIntegration.h"

using namespace std;  // for _1, _2, _3...
//using namespace std::placeholders;  // for _1, _2, _3...

enum PlasticIncrementMethod {
    GenericIntegrator, MidPoint, MidPointNR , ExplicitPredictorImplicitCorrector
};


template<unsigned int dim>
class HardeningLaw {
public:
    virtual double sig_y(const double &ep) const = 0;

    virtual double dsig_y(const double &ep) const = 0;

    virtual double
    h(const double &dlambda, const Tensor<2, dim> &tau, const double &ep) const = 0;
};

template<unsigned int dim>
class IsotropicHardeningLaw : public HardeningLaw<dim> {
public:

    double
    h(const double &dlambda, const Tensor<2, dim> &tau, const double &ep) const override {
        if (dlambda > 0)
            return 1.;
        else if (dlambda < 0)
            return -1.;
        else
            return 0;
    }

};

template<unsigned int dim>
class PerfectPlasticity : public IsotropicHardeningLaw<dim> {
public:
    PerfectPlasticity(double sig_y_init)
            : sig_y_init(sig_y_init) {};

    double sig_y(const double &ep) const override { return sig_y_init; };

    double dsig_y(const double &ep) const override { return 0; };

private:
    const double sig_y_init = 60;
//    double Y = 1;
};

template<unsigned int dim>
class LinearHardening : public IsotropicHardeningLaw<dim> {
public:
    LinearHardening(double sig_y_init, double H)
            : sig_y_init(sig_y_init), H(H) {};

    double sig_y(const double &ep) const override { return sig_y_init + H * ep; };

    double dsig_y(const double &ep) const override { return H; };

private:
    const double sig_y_init;
    const double H;
};


template<unsigned int dim>
class Voce : public IsotropicHardeningLaw<dim> {
public:
    Voce(double sig_y_init, double sig_inf, double del)
            : sig_y_init(sig_y_init), sig_inf(sig_inf), del(del) {};

    double sig_y(const double &ep) const override { return sig_inf - (sig_inf - sig_y_init) * exp(-del * ep); };

    double dsig_y(const double &ep) const override { return del * (sig_inf - sig_y_init) * exp(-del * ep); };

private:
    const double sig_y_init;
    const double sig_inf;
    const double del;
};

template<unsigned int dim>
class VoceWLinear : public IsotropicHardeningLaw<dim> {
public:
    VoceWLinear(double sig_y_init, double sig_inf, double del, double H)
            : sig_y_init(sig_y_init), sig_inf(sig_inf), del(del), H(H) {};

    double sig_y(const double &ep) const override {
        return sig_inf - (sig_inf - sig_y_init) * exp(-del * ep) + H * ep;
    };

    double dsig_y(const double &ep) const override { return del * (sig_inf - sig_y_init) * exp(-del * ep) + H; };

private:
    const double sig_y_init;
    const double sig_inf;
    const double del;
    const double H;
};

template<unsigned int dim>
class YieldSurface {
public:

    YieldSurface(HardeningLaw<dim> *hardening_law) : hardening_law(hardening_law) {};

    virtual double f(const Tensor<2, dim> &tau,
                     const double &ep) const = 0;

    virtual Tensor<2, dim> df_dtau(const Tensor<2, dim> &tau,
                                   const double &ep) const = 0;

    virtual double df_dxi(const Tensor<2, dim> &tau,
                          const double &ep) const = 0;


    const HardeningLaw<dim> *get_hardening_law() const { return hardening_law; };
    const HardeningLaw<dim> *hardening_law;
protected:

};

template<unsigned int dim>
class VonMises : public YieldSurface<dim> {
public:
    VonMises(HardeningLaw<dim> *hardening_law) : YieldSurface<dim>(hardening_law) {};

    double f(const Tensor<2, dim> &tau, const double &ep) const override {
        return (tau - Physics::Elasticity::StandardTensors<dim>::I * trace(tau) / 3.).norm()
               - sqrt(2. / 3.) * this->hardening_law->sig_y(ep);
    };

    Tensor<2, dim> df_dtau(const Tensor<2, dim> &tau, const double &ep) const override {
        Tensor<2, dim> t_bar = tau - Physics::Elasticity::StandardTensors<dim>::I * trace(tau) / 3.;
        return t_bar / t_bar.norm();
    };

    double df_dxi(const Tensor<2, dim> &tau, const double &ep) const override {
        return -sqrt(2. / 3.) * this->hardening_law->dsig_y(ep);
    };
};

template<unsigned int dim>
class FlowRule {
public:
    virtual Tensor<2, dim> T(const Tensor<2, dim> &tau, const double &ep) const = 0;
};

template<unsigned int dim>
class MaximumDissipation : public FlowRule<dim> {
public:
    MaximumDissipation() = default;

    Tensor<2, dim> T(const Tensor<2, dim> &tau, const double &ep) const override {
        Tensor<2, dim> t_bar = tau - Physics::Elasticity::StandardTensors<dim>::I * trace(tau) / 3.;
        return t_bar / t_bar.norm();
    };
};

template<unsigned int dim>
class PlasticityTheory {
public:
//    PlasticityTheory() = default;
    PlasticityTheory(YieldSurface<dim> *yield_surface, FlowRule<dim> *flow_rule)
            : yield_surface(yield_surface), flow_rule(flow_rule) {};

    const YieldSurface<dim> *yield_surface;
    const FlowRule<dim> *flow_rule;
};

//TODO fix return mapping algorithm and check tangent
template<unsigned int dim>
class RateIndependentPlasticity : public Material<dim> {
public:

    RateIndependentPlasticity(ElasticMaterial<dim> *e_law,
                              PlasticityTheory<dim> *p_theory,
                              PlasticIncrementMethod method = MidPoint)
            : Material<dim>(e_law->get_id()), elastic_law(e_law), plasticity_theory(p_theory), inc_method(method) {
        iota(range.begin(), range.end(), 0);
        throw NotImplemented("'RateIndependentPlasticity' has not been sufficiently implemented yet.");
    };

    void update_stress(const double &dt);

    void update_tangent(const double &dt) {
        throw NotImplemented("'update_stress' has not been implemented for 'RateIndependentPlasticity'. "
                             "Use 'update_stress_and_tangent' instead.");
    };

    void update_stress_and_tangent(const double &dt) override;

//    void increment(const double &dt,
//                   Tensor<4, dim> &c,
//                   Tensor<2, dim> &tau);
//
//
//    Tensor<4, dim> approximate_tangent(const double &dt);
//
//    void increment_old(const double &dt,
//                       Tensor<4, dim> &c,
//                       Tensor<2, dim> &tau);
//
//    void increment_old_old(const double &dt,
//                           Tensor<4, dim> &c,
//                           Tensor<2, dim> &tau);


//    void set_state(PlasticityState<dim> *state_ptr) { this->state = state_ptr; };

    typedef RateIndependentPlasticityState<dim> state_type;

    void set_state(StateBase<dim> *new_ptr) override { this->state = dynamic_cast<state_type *>(new_ptr); };

    StateBase<dim> *create_state() const override { return dynamic_cast<StateBase<dim> *>(new state_type()); }

private:
//    const ElasticMaterial<dim> *elastic_law;
    ElasticMaterial<dim> *elastic_law;
    const PlasticityTheory<dim> *plasticity_theory;
    const PlasticIncrementMethod inc_method;
    state_type *state;

    array<unsigned int, dim> range;
    const double alpha = 0.5;
    const double small_number = 1e-1;
//    const double small_number = 1e-6;

    TimeIntegration *integrator = new RungeKutta();
//    IntegrationMethod int_method = ExplicitPredictorImplicitCorrector;
//    IntegrationMethod int_method = MidPoint;

    void set_e_law_state() { elastic_law->set_state(state->get_elastic_component()); };

    template<class T>
    T mid_step(const T &in_n, const T &in_n1) const { return alpha * in_n1 + (1 - alpha) * in_n; };

//    void initialize_cross_yield();

    double initialize_cross_yield(const double &dt);

    static Tensor<2, dim> dtau_p(const Tensor<4, dim> &ce,
                                 const Tensor<2, dim> &tau,
                                 const Tensor<2, dim> &N,
                                 const Tensor<2, dim> &sym_N);

    static Tensor<2, dim> dtau_e(const Tensor<4, dim> &ce,
                                 const Tensor<2, dim> &tau,
                                 const Tensor<2, dim> &L,
                                 const Tensor<2, dim> &D);

    static Tensor<2, dim> dtau_e_inner_df_dtau(const Tensor<4, dim> &ce,
                                               const Tensor<2, dim> &tau,
                                               const Tensor<2, dim> &df_dtau);

    double dlambda(const Tensor<4, dim> &ce,
                   const Tensor<2, dim> &tau,
                   const double &ep,
                   const Tensor<2, dim> &N,
                   const Tensor<2, dim> &L,
                   const Tensor<2, dim> &D,
                   const double &dlambda);

    static Tensor<2, dim> sym(const Tensor<2, dim> &in) { return (in + transpose(in)) / 2.; };

    Tensor<2, dim> F_pn1(const double &delta_lambda,
                         const Tensor<2, dim> &T_n05) const;

    void mid_point_method(const double &dt);

    void mid_point_method_nr(const double &dt);

    double mid_point_r(const Tensor<2, dim> &T_nh,
                       const double &ep_n,
                       const Tensor<2, dim> &F_n1,
                       const double &delta_lambda);

    double mid_point_nr(const Tensor<2, dim> &T_n,
                        Tensor<2, dim> &T_n1,
                        const double &ep_n,
                        double &ep_n1,
                        Tensor<2, dim> &Fp_n1,
                        const Tensor<2, dim> &F_n1,
                        double &delta_lambda);

    SymmetricTensor<4, dim> tangent(const SymmetricTensor<4, dim> &ce,
                                    const Tensor<2, dim> &tau,
                                    const Tensor<2, dim> &N,
                                    const double &ep,
                                    const double &dlambda) const;


};

template<unsigned int dim>
void RateIndependentPlasticity<dim>::update_stress(const double &dt){
    state->update_elastic_component();
    set_e_law_state();
    elastic_law->update_stress_and_tangent(dt);

    double f = plasticity_theory->yield_surface->f(state->get_elastic_component()->tau_n1,
                                                   state->ep_n);

    if (f < 0) {
        state->tau_n1 = state->get_elastic_component()->tau_n1;
        state->c_n1 = state->get_elastic_component()->c_n1;
        state->plastic_flow_n1 = false;
        state->f_n1 = f;
    } else {
        state->plastic_flow_n1 = true;
        auto *working_state = new state_type(state);
        auto *actual_state = state;
        state = working_state;
        double dt_plastic = dt;
        if (not state->plastic_flow_n) {
            dt_plastic = initialize_cross_yield(dt);
        }

        switch (inc_method) {
            case GenericIntegrator:
                throw NotImplemented(
                        "Chosen plastic increment method 'GenericIntegrator' has not been implemented yet.");
                break;
            case MidPoint:
                mid_point_method(dt_plastic);
                break;
            case MidPointNR:
                mid_point_method_nr(dt_plastic);
                break;
            case ExplicitPredictorImplicitCorrector:
                throw NotImplemented(
                        "Chosen plastic increment method 'ExplicitPredictorImplicitCorrector' has not been implemented yet.");
                break;
            default:
                throw NotImplemented("Chosen plastic increment method has not been implemented yet.");
        }

        actual_state->copy_new_values(working_state);
        delete working_state;
        state = actual_state;
    }

    state->initial_nr_iteration = false;

}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::update_stress_and_tangent(const double &dt) {
    update_stress(dt);
    Material<dim>::approximate_tangent(dt, state);
}

template<unsigned int dim>
Tensor<2, dim> RateIndependentPlasticity<dim>::dtau_e(const Tensor<4, dim> &ce,
                                                      const Tensor<2, dim> &tau,
                                                      const Tensor<2, dim> &L,
                                                      const Tensor<2, dim> &D) {
    return double_contract<2, 0, 3, 1>(ce, D) + L * tau + tau * transpose(L);
}

template<unsigned int dim>
Tensor<2, dim>
RateIndependentPlasticity<dim>::dtau_p(const Tensor<4, dim> &ce,
                                       const Tensor<2, dim> &tau,
                                       const Tensor<2, dim> &N,
                                       const Tensor<2, dim> &sym_N) {
    return double_contract<2, 0, 3, 1>(ce, sym_N) + N * tau + tau * transpose(N);
}

template<unsigned int dim>
Tensor<2, dim> RateIndependentPlasticity<dim>::dtau_e_inner_df_dtau(const Tensor<4, dim> &ce,
                                                                    const Tensor<2, dim> &tau,
                                                                    const Tensor<2, dim> &df_dtau) {
    return double_contract<0, 0, 1, 1>(df_dtau, ce) + df_dtau * tau + tau * df_dtau;
}

template<unsigned int dim>
double RateIndependentPlasticity<dim>::dlambda(const Tensor<4, dim> &ce,
                                               const Tensor<2, dim> &tau,
                                               const double &ep,
                                               const Tensor<2, dim> &N,
                                               const Tensor<2, dim> &L,
                                               const Tensor<2, dim> &D,
                                               const double &dlambda) {
    Tensor<2, dim> sym_N = sym(N);
    Tensor<2, dim> df_dtau = plasticity_theory->yield_surface->df_dtau(tau, ep);
    double df_dep = plasticity_theory->yield_surface->df_dxi(tau, ep);
    double h = plasticity_theory->yield_surface->hardening_law->h(dlambda, tau, ep);
    return scalar_product(df_dtau, dtau_e(ce, tau, L, D)) /
           (scalar_product(df_dtau, dtau_p(ce, tau, N, sym_N)) - df_dep * h);
}

template<unsigned int dim>
Tensor<2, dim> RateIndependentPlasticity<dim>::F_pn1(const double &delta_lambda,
                                                     const Tensor<2, dim> &T_n05) const {
    return invert(this->I - alpha * delta_lambda * T_n05) *
           (this->I + (1 - alpha) * delta_lambda * T_n05) * state->Fp_n;
}

template<unsigned int dim>
void RateIndependentPlasticity<dim>::mid_point_method(const double &dt) {


    ElasticState<dim> *working_state = new ElasticState<dim>(state->get_elastic_component());
    elastic_law->set_state(working_state);

    SymmetricTensor<4, dim> ce_n, ce_nh, ce_n1;
    Tensor<2, dim> tau_n, F_n, Fe_n, Fp_n, T_n, dFp_n, N_n;
    Tensor<2, dim> tau_nh, F_nh, Fe_nh, Fp_nh, T_nh, N_nh;
    Tensor<2, dim> tau_n1, F_n1, Fe_n1, Fp_n1, T_n1, dFp_n1, N_n1;
    Tensor<2, dim> dFdt, D, L;

    double dlambda_n, ep_n;
    double dlambda_nh, ep_nh;
    double dlambda_n1, ep_n1;
    double delta_lambda;

//    tau_n = state->tau_n;
    F_n = state->F_n;
    Fp_n = state->Fp_n;
    Fe_n = state->get_elastic_component()->F_n;
    T_n = state->T_n;
    N_n = Fe_n * T_n * invert(Fe_n);
    dlambda_n = state->dlam_n;
    dFp_n = state->dFp_n;
    ep_n = state->ep_n;

    working_state->F_n1 = Fe_n;
    elastic_law->update_stress_and_tangent(dt);
    tau_n = working_state->tau_n1;
    ce_n = working_state->c_n1;

    F_n1 = state->F_n1;
    F_nh = mid_step(F_n, F_n1);
    dFdt = (F_n1 - F_n) / dt;
    L = dFdt * invert(F_nh);
    D = sym(L);


//  Initial guesses
    if (state->initial_nr_iteration) {
//        ep_n1 = ep_n + dt * dlambda_n * plasticity_theory->yield_surface->hardening_law->h(dlambda_n, tau_n, ep_n);
        ep_n1 = ep_n + dt * fabs(dlambda_n);
        Fp_n1 = dFp_n * dt + Fp_n;
        Fe_n1 = F_n1 * invert(Fp_n1);
        working_state->F_n1 = Fe_n1;
        elastic_law->update_stress_and_tangent(dt);
        tau_n1 = working_state->tau_n1;
        ce_n1 = working_state->c_n1;
        T_n1 = plasticity_theory->flow_rule->T(tau_n1, ep_n1);
        N_n1 = Fe_n1 * T_n1 * invert(Fe_n1);
    } else {
        ep_n1 = state->ep_n1;
        Fp_n1 = state->Fp_n1;
        Fe_n1 = state->elastic_component->F_n1;
        working_state->F_n1 = Fe_n1;
        working_state->tau_n1 = state->elastic_component->tau_n1;
        working_state->c_n1 = state->elastic_component->c_n1;
        tau_n1 = working_state->tau_n1;
        ce_n1 = working_state->c_n1;
        T_n1 = state->T_n1;
        N_n1 = Fe_n1 * T_n1 * invert(Fe_n1);
    }

    double f = plasticity_theory->yield_surface->f(tau_n1, ep_n1);

    while (fabs(f) / tau_n1.norm() > small_number) {

        ce_nh = mid_step(ce_n, ce_n1);
        tau_nh = mid_step(tau_n, tau_n1);
        ep_nh = mid_step(ep_n, ep_n1);
        N_nh = mid_step(N_n, N_n1);
        T_nh = mid_step(T_n, T_n1);

        delta_lambda = dt * dlambda(ce_nh, tau_nh, ep_nh, N_nh, L, D, dlambda_n);
        Fp_n1 = F_pn1(delta_lambda, T_nh);

//        ep_n1 = ep_n + delta_lambda * plasticity_theory->yield_surface->hardening_law->h(delta_lambda, tau_nh, ep_nh);
        ep_n1 = ep_n + fabs(delta_lambda);
        Fe_n1 = F_n1 * invert(Fp_n1);
        working_state->F_n1 = Fe_n1;
        elastic_law->update_stress_and_tangent(dt);
        tau_n1 = working_state->tau_n1;
        ce_n1 = working_state->c_n1;
        T_n1 = plasticity_theory->flow_rule->T(tau_n1, ep_n1);
        N_n1 = Fe_n1 * T_n1 * invert(Fe_n1);

        f = plasticity_theory->yield_surface->f(tau_n1, ep_n1);
    }


    state->Fp_n1 = Fp_n1;
    state->T_n1 = T_n1;
    state->ep_n1 = ep_n1;
    state->dFp_n1 = (Fp_n1 - Fp_n) / dt;
    state->dlam_n1 = delta_lambda / dt;
    state->f_n1 = f;

    state->tau_n1 = tau_n1;
    state->elastic_component->c_n1 = ce_n1;
    state->c_n1 = tangent(ce_n1, tau_n1, N_n1, state->ep_n1, state->dlam_n1);

    state->update_elastic_component();
    elastic_law->set_state(state->get_elastic_component());
    elastic_law->update_stress_and_tangent(dt);

    delete working_state;
}


template<unsigned int dim>
void RateIndependentPlasticity<dim>::mid_point_method_nr(const double &dt) {
    auto *working_state = new ElasticState<dim>(state->get_elastic_component());
    elastic_law->set_state(working_state);

    SymmetricTensor<4, dim> ce_n, ce_nh, ce_n1;
    Tensor<2, dim> tau_n, F_n, Fe_n, Fp_n, T_n, dFp_n, N_n;
    Tensor<2, dim> tau_nh, F_nh, Fe_nh, Fp_nh, T_nh, N_nh;
    Tensor<2, dim> tau_n1, F_n1, Fe_n1, Fp_n1, T_n1, dFp_n1, N_n1;
    Tensor<2, dim> dFdt, D, L;

    double dlambda_n, ep_n;
    double dlambda_nh, ep_nh;
    double dlambda_n1, ep_n1;
    double delta_lambda;

//    tau_n = state->tau_n;
    F_n = state->F_n;
    Fp_n = state->Fp_n;
    Fe_n = state->get_elastic_component()->F_n;
    T_n = state->T_n;
    N_n = Fe_n * T_n * invert(Fe_n);
    dlambda_n = state->dlam_n;
    dFp_n = state->dFp_n;
    ep_n = state->ep_n;

    working_state->F_n1 = Fe_n;
    elastic_law->update_stress_and_tangent(dt);
    tau_n = working_state->tau_n1;
    ce_n = working_state->c_n1;

    F_n1 = state->F_n1;
    F_nh = mid_step(F_n, F_n1);
    dFdt = (F_n1 - F_n) / dt;
    L = dFdt * invert(F_nh);
    D = sym(L);


//  Initial guesses
    if (state->initial_nr_iteration) {
//        ep_n1 = ep_n + dt * dlambda_n * plasticity_theory->yield_surface->hardening_law->h(dlambda_n, tau_n, ep_n);
        ep_n1 = ep_n + dt * fabs(dlambda_n);
        Fp_n1 = dFp_n * dt + Fp_n;
        Fe_n1 = F_n1 * invert(Fp_n1);
        working_state->F_n1 = Fe_n1;
        elastic_law->update_stress_and_tangent(dt);
        tau_n1 = working_state->tau_n1;
        ce_n1 = working_state->c_n1;
        T_n1 = plasticity_theory->flow_rule->T(tau_n1, ep_n1);
        N_n1 = Fe_n1 * T_n1 * invert(Fe_n1);
    } else {
        ep_n1 = state->ep_n1;
        Fp_n1 = state->Fp_n1;
        Fe_n1 = state->elastic_component->F_n1;
        working_state->F_n1 = Fe_n1;
        working_state->tau_n1 = state->elastic_component->tau_n1;
        working_state->c_n1 = state->elastic_component->c_n1;
        tau_n1 = working_state->tau_n1;
        ce_n1 = working_state->c_n1;
        T_n1 = state->T_n1;
        N_n1 = Fe_n1 * T_n1 * invert(Fe_n1);
    }

    ce_nh = mid_step(ce_n, ce_n1);
    tau_nh = mid_step(tau_n, tau_n1);
    ep_nh = mid_step(ep_n, ep_n1);
    N_nh = mid_step(N_n, N_n1);
    T_nh = mid_step(T_n, T_n1);

    delta_lambda = dt * dlambda(ce_nh, tau_nh, ep_nh, N_nh, L, D, dlambda_n);
    Fp_n1 = F_pn1(delta_lambda, T_nh);

    double f = mid_point_nr(T_n, T_n1, ep_n, ep_n1, Fp_n1, F_n1, delta_lambda);

    state->Fp_n1 = Fp_n1;
    state->T_n1 = T_n1;
    state->ep_n1 = ep_n1;
    state->dFp_n1 = (Fp_n1 - Fp_n) / dt;
    state->dlam_n1 = delta_lambda / dt;
    state->f_n1 = f;

    state->update_elastic_component();
    elastic_law->set_state(state->get_elastic_component());
    elastic_law->update_stress_and_tangent(dt);

    state->tau_n1 = state->elastic_component->tau_n1;

    N_n1 = state->elastic_component->F_n1 * T_n1 * invert(state->elastic_component->F_n1);

    state->c_n1 = tangent(state->elastic_component->c_n1,
                          state->tau_n1,
                          N_n1,
                          state->ep_n1,
                          state->dlam_n1);

    delete working_state;
}


template<unsigned int dim>
double RateIndependentPlasticity<dim>::mid_point_nr(const Tensor<2, dim> &T_n,
                                                    Tensor<2, dim> &T_n1,
                                                    const double &ep_n,
                                                    double &ep_n1,
                                                    Tensor<2, dim> &Fp_n1,
                                                    const Tensor<2, dim> &F_n1,
                                                    double &delta_lambda) {
    auto *working_state = new ElasticState<dim>(state->get_elastic_component());
    elastic_law->set_state(working_state);
    double eps = 1e-7*delta_lambda;

    Tensor<2, dim> T_nh = mid_step(T_n, T_n1);
    double r = mid_point_r(T_nh, ep_n, F_n1, delta_lambda);
    double dr_dlam, rh;

    while (fabs(r) > small_number) {
        rh = mid_point_r(T_nh, ep_n, F_n1, delta_lambda + eps);
        dr_dlam = (rh - r) / eps;
        elastic_law->set_state(working_state);
        delta_lambda -= dr_dlam * r;
        Fp_n1 = F_pn1(delta_lambda, T_nh);
        ep_n1 = ep_n + fabs(delta_lambda);
        working_state->F_n1 = F_n1 * invert(Fp_n1);
        elastic_law->update_stress(0);
        T_n1 = plasticity_theory->flow_rule->T(working_state->tau_n1, ep_n1);
        T_nh = mid_step(T_n, T_n1);
        r = mid_point_r(T_nh, ep_n, F_n1, delta_lambda);
    }

    delete working_state;

    return r;
}

template<unsigned int dim>
double RateIndependentPlasticity<dim>::mid_point_r(const Tensor<2, dim> &T_nh,
                                                   const double &ep_n,
                                                   const Tensor<2, dim> &F_n1,
                                                   const double &delta_lambda) {

    auto *working_state = new ElasticState<dim>(state->get_elastic_component());
    elastic_law->set_state(working_state);

    Tensor<2, dim> Fp_n1 = F_pn1(delta_lambda, T_nh);
    double ep_n1 = ep_n + fabs(delta_lambda);
    Tensor<2, dim> Fe_n1 = F_n1 * invert(Fp_n1);
    working_state->F_n1 = Fe_n1;
    elastic_law->update_stress(0);
    double f = plasticity_theory->yield_surface->f(working_state->tau_n1, ep_n1);

    elastic_law->set_state(state->get_elastic_component());
    delete working_state;

    return f;
}


template<unsigned int dim>
SymmetricTensor<4, dim>
RateIndependentPlasticity<dim>::tangent(const SymmetricTensor<4, dim> &ce,
                                        const Tensor<2, dim> &tau,
                                        const Tensor<2, dim> &N,
                                        const double &ep,
                                        const double &dlambda) const {
    Tensor<2, dim> sym_N = sym(N);
    Tensor<2, dim> df_dtau = plasticity_theory->yield_surface->df_dtau(tau, ep);

    SymmetricTensor<2, dim> dtau_p_val = symmetrize(dtau_p(ce, tau, N, sym_N));
    SymmetricTensor<2, dim> dtau_e_val = symmetrize(dtau_e_inner_df_dtau(ce, tau, df_dtau));

    double df_dxi = plasticity_theory->yield_surface->df_dxi(tau, ep);
    double h = plasticity_theory->yield_surface->hardening_law->h(dlambda, tau, ep);

    return ce - outer_product(dtau_p_val, dtau_e_val) / (scalar_product(df_dtau, dtau_p_val) - df_dxi * h);
}


template<unsigned int dim>
double RateIndependentPlasticity<dim>::initialize_cross_yield(const double &dt) {
    Tensor<2, dim> dF = state->F_n1 - state->F_n;
    double fraction = 0.5;
    double in_base = 0.5;


    Tensor<2, dim> Fp_n_inv = invert(state->Fp_n);

    state->elastic_component->F_n1 = (state->F_n + dF * fraction) * Fp_n_inv;
    elastic_law->update_stress(dt);
    Tensor<2, dim> tau = state->elastic_component->tau_n1;
    double ep_n = state->ep_n;

    double f = plasticity_theory->yield_surface->f(tau, ep_n);
    double counter = 2;
    double fraction_prev = fraction + 1;
    double eps = 1e-4;
//    while (fabs(f) > small_number) {
    while (fabs(fraction_prev - fraction) > eps) {
        fraction_prev = fraction;
        if (f > 0)
            fraction -= pow(in_base, counter);
        else
            fraction += pow(in_base, counter);

//        Ffrac = (state->F_n + dF * fraction) * invert(state->Fp_n);

        state->elastic_component->F_n1 = (state->F_n + dF * fraction) * Fp_n_inv;
        elastic_law->update_stress(dt);
        tau = state->elastic_component->tau_n1;

//        tau = elastic_law->tau(Ffrac);
        f = plasticity_theory->yield_surface->f(tau, ep_n);
        counter++;
    }
    double dt_plastic = dt * (1 - fraction);

    state->F_n = state->F_n + dF * fraction;
    state->tau_n = tau;
    state->T_n = plasticity_theory->flow_rule->T(tau, ep_n);

    elastic_law->update_stress_and_tangent(dt);

    Tensor<2, dim> F_nh = mid_step(state->F_n, state->F_n1);
    Tensor<2, dim> dFdt = (state->F_n1 - state->F_n) / dt;
    Tensor<2, dim> L = dFdt * invert(F_nh);
    Tensor<2, dim> D = sym(L);

    Tensor<2, dim> N = state->elastic_component->F_n1 * state->T_n * invert(state->elastic_component->F_n1);

    double dlambda_dir = scalar_product(L, state->T_n);

    state->dlam_n = dlambda(state->elastic_component->c_n1, tau, ep_n, N, L, D, dlambda_dir);

    return dt_plastic;
}

#endif //SOLVER_RATEINDEPENDTPLASTICITY_H
