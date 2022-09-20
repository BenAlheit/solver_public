//
// Created by alhei on 2022/09/20.
//

#ifndef SOLVER_EXPLICITRATEDEPENDENTCRYSTALPLASTICITY_H
#define SOLVER_EXPLICITRATEDEPENDENTCRYSTALPLASTICITY_H

#include "Material.h"
#include "MaterialState.h"
#include "math.h"

inline double to_degrees(const double & radians) {
    return radians * (180.0 / M_PI);
}

inline double to_radians(const double & degrees) {
    return degrees * (M_PI / 180.0);
}

using namespace std;
using namespace dealii;

template<unsigned int dim>
class CrystalPlasticityState : public KronnerDecomp<dim> {
public:
    CrystalPlasticityState(const unsigned int &n,
                           const vector<double> &ref_euler_angles = vector<double>(0, dim))
            : KronnerDecomp<dim>(), n(n), Fp_n(Physics::Elasticity::StandardTensors<dim>::I),
              Fp_n1(Physics::Elasticity::StandardTensors<dim>::I), ref_s(n), ref_m(n), nu_n(n), nu_n1(n), ta_n(n),
              ta_n1(n), H_n(n), H_n1(n), ref_euler_angles(ref_euler_angles) {};


    CrystalPlasticityState(CrystalPlasticityState<dim> *to_cpy)
            : KronnerDecomp<dim>(to_cpy), n(to_cpy->n), Fp_n(to_cpy->Fp_n), Fp_n1(to_cpy->Fp_n1), ref_s(to_cpy->ref_s),
              ref_m(to_cpy->ref_m), nu_n(to_cpy->nu_n), nu_n1(to_cpy->nu_n1), ta_n(to_cpy->ta_n), ta_n1(to_cpy->ta_n1),
              H_n(to_cpy->H_n), H_n1(to_cpy->H_n1), ref_euler_angles(to_cpy->ref_euler_angles) {};


    virtual StateBase<dim> *
    copy() override { return dynamic_cast<StateBase<dim> *>(new CrystalPlasticityState<dim>(n, ref_euler_angles)); };

    void copy(CrystalPlasticityState<dim> *to_cpy) {
        KronnerDecomp<dim>::copy(to_cpy);

        this->n = to_cpy->n;
        this->ref_euler_angles = to_cpy->ref_euler_angles;
        this->ref_s = to_cpy->ref_s;
        this->ref_m = to_cpy->ref_m;

        this->Fp_n = to_cpy->Fp_n;
        this->nu_n = to_cpy->nu_n;
        this->ta_n = to_cpy->ta_n;
        this->H_n = to_cpy->H_n;

        this->Fp_n1 = to_cpy->Fp_n1;
        this->nu_n1 = to_cpy->nu_n1;
        this->ta_n1 = to_cpy->ta_n1;
        this->H_n1 = to_cpy->H_n1;

    };


    void update_elastic_component() override {
        this->elastic_component->F_n1 = this->F_n1 * invert(Fp_n1);
    };


    void update() override {
        update_elastic_component();
        this->elastic_component->update();
        StateBase<dim>::update();

        Fp_n = Fp_n1;
        nu_n = nu_n1;
        ta_n = ta_n1;
        H_n = H_n1;
    };

    void reset() override {
        Fp_n1 = Fp_n;
        nu_n1 = nu_n;
        ta_n1 = ta_n;
        H_n1 = H_n;

        update_elastic_component();
        this->elastic_component->reset();
        StateBase<dim>::reset();
    };

    void copy_new_values(CrystalPlasticityState<dim> *to_cpy) {
        Fp_n1 = to_cpy->Fp_n1;
        nu_n1 = to_cpy->nu_n1;
        ta_n1 = to_cpy->ta_n1;
        H_n1 = to_cpy->H_n1;
    };

    double scalar_output(ScalarOutputFlag flag) override {
        return StateBase<dim>::scalar_output(flag);
    };

    Tensor<1, dim> vector_output(VectorOutputFlag flag) override {
        return StateBase<dim>::vector_output(flag);
    };

    Tensor<2, dim> tensor_output(TensorOutputFlag flag) override {
        return StateBase<dim>::tensor_output(flag);
    };

    double n_scalar_output(const nScalarOutputFlag & flag, const unsigned int& i) override {
        switch (flag) {
            case nScalarOutputFlag::H:
                return H_n1.at(i);
            case nScalarOutputFlag::ta:
                return ta_n1.at(i);
            case nScalarOutputFlag::nu:
                return nu_n1.at(i);
            default:
                return StateBase<dim>::n_scalar_output(flag, i);
        }
    };

    Tensor<1, dim> n_vector_output(const nVectorOutputFlag & flag, const unsigned int& i) override {
        switch (flag) {
            case nVectorOutputFlag::M:
                return Fp_n1 * ref_m.at(i);
            case nVectorOutputFlag::S:
                return Fp_n1 * ref_s.at(i);
            default:
                return StateBase<dim>::n_vector_output(flag, i);
        }
    };

    Tensor<2, dim> n_tensor_output(const nTensorOutputFlag & flag, const unsigned int& i) override {
        switch (flag) {
            case nTensorOutputFlag::SYS:
                return Fp_n1 * ref_sys.at(i);
            default:
                return StateBase<dim>::n_tensor_output(flag, i);
        }
    };

    const unsigned int n;

    Tensor<2, dim> Fp_n;
    Tensor<2, dim> Fp_n1;

    vector<Tensor<1, dim>> ref_s;
    vector<Tensor<1, dim>> ref_m;
    vector<Tensor<2, dim>> ref_sys;

    Tensor<2, dim> ref_rotation_matrix;

    vector<double> ta_n;
    vector<double> ta_n1;
    vector<double> nu_n;
    vector<double> nu_n1;
    vector<double> H_n;
    vector<double> H_n1;

    const vector<double> ref_euler_angles;

protected:

    void initialize(){
        const SymmetricTensor<2, dim> & I_ref = Physics::Elasticity::StandardTensors<dim>::I;
        Tensor<2, dim> Rx(I_ref), Ry(I_ref), Rz(I_ref);

        Rx[1][1] = cos(to_radians(ref_euler_angles.at(0)));
        Rx[2][2] = cos(to_radians(ref_euler_angles.at(0)));
        Rx[1][2] = -sin(to_radians(ref_euler_angles.at(0)));
        Rx[2][1] = sin(to_radians(ref_euler_angles.at(0)));

        Ry[0][0] = cos(to_radians(ref_euler_angles.at(1)));
        Ry[2][2] = cos(to_radians(ref_euler_angles.at(1)));
        Ry[0][2] = sin(to_radians(ref_euler_angles.at(1)));
        Ry[2][0] = -sin(to_radians(ref_euler_angles.at(1)));


        Rx[0][0] = cos(to_radians(ref_euler_angles.at(2)));
        Rx[1][1] = cos(to_radians(ref_euler_angles.at(2));
        Rx[0][1] = -sin(to_radians(ref_euler_angles.at(2)));
        Rx[1][0] = sin(to_radians(ref_euler_angles.at(2)));

        ref_rotation_matrix = Rz * Ry * Rx;

        for (unsigned int alpha = 0; alpha < n ; ++alpha) {
            ref_s.at(alpha) = ref_rotation_matrix * ref_s.at(alpha);
            ref_m.at(alpha) = ref_rotation_matrix * ref_m.at(alpha);
            ref_sys.at(alpha) = outer_product(ref_s.at(alpha), ref_m.at(alpha));
        }
    }

};

template<unsigned int dim>
class SingleSlipCrystalPlasticityState : CrystalPlasticityState<dim>{
//    TODO
//    TODO
//    TODO
//    TODO
//    TODO
};

template<unsigned int dim>
class ExplicitRateCrystalDependentPlasticity : public Material<dim> {
public:
    typedef CrystalPlasticityState<dim> state_type;

    ExplicitRateCrystalDependentPlasticity(ElasticMaterial<dim> *e_law,
                                           state_type *example_state,
                                           const double &sig_y,
                                           const double &mu,
                                           const double &m,
                                           const double &q)
            : Material<dim>(e_law->get_id()), elastic_law(e_law), state(example_state), sig_y(sig_y), mu(mu), m(m),
              q(q), n(example_state->n), sys_range(example_state->n) { iota(sys_range.begin(), sys_range.end(), 0); };

    void update_stress(const double &dt) override {
        throw NotImplemented("'update_stress' has not been implemented for 'ExplicitRateDependentPlasticity'. "
                             "Use 'update_stress_and_tangent' instead.");
    };

    void update_tangent(const double &dt) override {
        throw NotImplemented("'update_tangent' has not been implemented for 'ExplicitRateDependentPlasticity'. "
                             "Use 'update_stress_and_tangent' instead.");
    };

    void update_stress_and_tangent(const double &dt) override;


    void set_state(StateBase<dim> *new_ptr) override { this->state = dynamic_cast<state_type *>(new_ptr); };

    StateBase<dim> *create_state() const override {
        return dynamic_cast<StateBase<dim> *>(new state_type(state->n,
                                                             state->ref_euler_angles));
    }

private:
    const double sig_y, mu, m, q;
    state_type *state;
    const unsigned int n;
    array<unsigned int, dim> range;
    vector<unsigned int> sys_range;
    ElasticMaterial<dim> *elastic_law;

    double h(const double &s) { return 1; };

    unsigned int is_coplanar(const unsigned int &alpha,
                             const unsigned int &beta) {
        return (unsigned int) almost_equals(fabs(state->ref_m.at(alpha) * state->ref_m.at(beta)), 1);
    };

};

template<unsigned int dim>
void ExplicitRateCrystalDependentPlasticity<dim>::update_stress_and_tangent(const double &dt) {
    if (almost_equals(state->H_n1.at(0), state->H_n.at(0))) {
        Tensor<2, dim> Lp, M, FeT, FeT_inv;
        double coplanar;
        FeT = transpose(state->elastic_component->F_n);
        FeT_inv = invert(FeT);
        M = FeT * state->tau_n * FeT_inv;
        for (const auto &alpha: sys_range) {
            state->ta_n1.at(alpha) = state->ref_s.at(alpha) * (M * state->ref_m.at(alpha));
            if (fabs(state->ta_n1.at(alpha)) > 0) {
                state->nu_n1.at(alpha) = (fabs(state->ta_n1.at(alpha)) / (state->ta_n1.at(alpha) * mu))
                                         * pow(state->ta_n1.at(alpha) / sig_y, 1. / m);
            }
            Lp += state->nu_n1.at(alpha) * state->ref_sys.at(alpha);
        }

        for (const auto &alpha: sys_range) {
            state->H_n1.at(alpha) = state->H_n.at(alpha);
            for (const auto &beta: sys_range) {
                coplanar = is_coplanar(alpha, beta);
                state->H_n1.at(alpha) += dt * (coplanar + q * (1 - coplanar))
                                         * h(state->H_n.at(beta))
                                         * fabs(state->nu_n1.at(beta));
            }
        }

        state->Fp_n1 = state->Fp_n + dt * Lp * state->Fp_n;
    }

    state->update_elastic_component();

    elastic_law->set_state(state->elastic_component);
    elastic_law->update_stress_and_tangent(dt);

    state->tau_n1 = state->elastic_component->tau_n1;
    state->c_n1 = state->elastic_component->c_n1;
}

#endif //SOLVER_EXPLICITRATEDEPENDENTCRYSTALPLASTICITY_H
