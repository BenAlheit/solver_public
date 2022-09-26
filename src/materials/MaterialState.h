//
// Created by alhei on 2022/08/29.
//

#ifndef SOLVER_MATERIALSTATE_H
#define SOLVER_MATERIALSTATE_H

#include <deal.II/base/tensor.h>
#include <string>
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include "../Exceptions.h"


#include "OutputFlags.h"

using namespace dealii;
using namespace OutputFlags;

template<unsigned int dim>
class StateBase {
public:
    StateBase()
            : F_n(Physics::Elasticity::StandardTensors<dim>::I), F_n1(Physics::Elasticity::StandardTensors<dim>::I) {};

    explicit StateBase(StateBase<dim> *to_cpy)
            : F_n(to_cpy->F_n), F_n1(to_cpy->F_n1), tau_n(to_cpy->tau_n), tau_n1(to_cpy->tau_n1), c_n(to_cpy->c_n),
              c_n1(to_cpy->c_n1) {};

    virtual void update() {
        tau_n = tau_n1;
        F_n = F_n1;
        c_n = c_n1;
    };

    virtual void reset() {
        tau_n = tau_n1;
        F_n1 = F_n;
        c_n1 = c_n;
    };

    virtual StateBase<dim> *copy() { return new StateBase<dim>(); };

    virtual void copy(StateBase<dim> *to_cpy) {
        this->F_n = to_cpy->F_n;
        this->F_n1 = to_cpy->F_n1;
        this->tau_n = to_cpy->tau_n;
        this->tau_n1 = to_cpy->tau_n1;
        this->c_n = to_cpy->c_n;
        this->c_n1 = to_cpy->c_n1;
    };

    virtual double scalar_output(ScalarOutputFlag flag) {
        switch (flag) {
            case ScalarOutputFlag::P:
                return trace(tau_n1) / 3.;
            case ScalarOutputFlag::J:
                return determinant(F_n1);
            default:
                throw NotImplemented(
                        "scalar_output is not implemented for flag '" + to_string(flag) + "' for this material state.");
        }
    };

    virtual Tensor<1, dim> vector_output(VectorOutputFlag flag) {
        switch (flag) {
            case VectorOutputFlag::NS1:
                return get_tau_eig(0); // TODO This is a very inefficient way of managing eigenvalue output (fix)
            case VectorOutputFlag::NS2:
                return get_tau_eig(1);
            case VectorOutputFlag::NS3:
                return get_tau_eig(2);
            case VectorOutputFlag::NB1:
                return get_B_eig(0);
            case VectorOutputFlag::NB2:
                return get_B_eig(1);
            case VectorOutputFlag::NB3:
                return get_B_eig(2);
            default:
                throw NotImplemented(
                        "vector_output is not implemented for flag '" + to_string<dim>(flag).at(0) + "' for this material state.");
        }
    };

    virtual Tensor<2, dim> tensor_output(TensorOutputFlag flag) {
        switch (flag) {
            case TensorOutputFlag::F:
                return F_n1;
            case TensorOutputFlag::STRESS:
                return tau_n1;
            case TensorOutputFlag::STRAIN:
                return 0.5 * (transpose(F_n1) * F_n1 - Physics::Elasticity::StandardTensors<dim>::I);
            case TensorOutputFlag::FirstPiolaStress:
                return tau_n1 * invert(transpose(F_n1));
            default:
                throw NotImplemented(
                        "tensor_output is not implemented for flag '" + to_string<dim>(flag).at(0) + "' for this material state.");
        }
    };

    virtual double n_scalar_output(const nScalarOutputFlag & flag, const unsigned int& i){
        throw NotImplemented(
                "'n_scalar_output' is not implemented for flag '" + to_string(flag, i) + "' for this material state.");
    };

    virtual Tensor<1, dim> n_vector_output(const nVectorOutputFlag & flag, const unsigned int& i){
        throw NotImplemented(
                "'n_vector_output' is not implemented for flag '" + to_string<dim>(flag, i).at(0) + "' for this material state.");
    };

    virtual Tensor<2, dim> n_tensor_output(const nTensorOutputFlag & flag, const unsigned int& i){
        throw NotImplemented(
                "'n_tensor_output' is not implemented for flag '" + to_string<dim>(flag, i).at(0) + "' for this material state.");
    };

    Tensor<2, dim> F_n;
    Tensor<2, dim> F_n1;

    Tensor<2, dim> tau_n;
    Tensor<2, dim> tau_n1;

    SymmetricTensor<4, dim> c_n;
    SymmetricTensor<4, dim> c_n1;

private:
    Tensor<1, dim> get_tau_eig(unsigned int comp) {
        SymmetricTensor<2, dim> tau_sym = symmetrize(tau_n1);
        const auto eigs = eigenvectors(tau_sym);
        return eigs[comp].first * eigs[comp].second;
    };

    Tensor<1, dim> get_B_eig(unsigned int comp) {
        SymmetricTensor<2, dim> B = symmetrize(F_n1 * transpose(F_n1));
        const auto eigs = eigenvectors(B);
        return eigs[comp].first * eigs[comp].second;
    };

};

template<unsigned int dim>
class ElasticState : public StateBase<dim> {
public:
    ElasticState() : StateBase<dim>() {};

    ElasticState(ElasticState<dim> *to_cpy) : StateBase<dim>(to_cpy) {};

    StateBase<dim> *copy() override { return dynamic_cast<StateBase<dim> *>(new ElasticState<dim>()); };

    double elastic_strain_energy_n;
    double elastic_strain_energy_n1;

    void update() override {
        StateBase<dim>::update();
        elastic_strain_energy_n = elastic_strain_energy_n1;
    };


    double scalar_output(ScalarOutputFlag flag) override {
        switch (flag) {
            case ScalarOutputFlag::ELASTIC_STRAIN_ENERGY:
                return elastic_strain_energy_n1;
            default:
                return StateBase<dim>::scalar_output(flag);
        }
    };

};

template<unsigned int dim>
class ViscoelasticState : public ElasticState<dim> {
public:
    explicit ViscoelasticState(const unsigned int & n)
    : ElasticState<dim>()
    , n(n)
    , tau_v_n(vector<Tensor<2, dim>>(n))
    , tau_v_n1(vector<Tensor<2, dim>>(n))
    {};

    explicit ViscoelasticState(ViscoelasticState<dim> *to_cpy)
    : ElasticState<dim>(to_cpy)
    , n(to_cpy->n)
    , tau_v_n(to_cpy->tau_v_n)
    , tau_v_n1(to_cpy->tau_v_n1)
    , tau_inf_n(to_cpy->tau_inf_n)
    , tau_inf_n1(to_cpy->tau_inf_n1)
    {};

    StateBase<dim> * copy() override { return dynamic_cast<StateBase<dim> *>(new ViscoelasticState<dim>(n)); };

    void copy(ViscoelasticState<dim> *to_cpy) {
        ElasticState<dim>::copy(to_cpy);
        this->n = to_cpy->n;
        this->tau_v_n = to_cpy->tau_v_n;
        this->tau_v_n1 = to_cpy->tau_v_n1;
        this->tau_inf_n = to_cpy->tau_inf_n;
        this->tau_inf_n1 = to_cpy->tau_inf_n1;
    };

    void update() override {
        ElasticState<dim>::update();
        tau_v_n = tau_v_n1;
        tau_inf_n = tau_inf_n1;
    };

    void reset() override {
        ElasticState<dim>::reset();
        tau_v_n1 = tau_v_n;
        tau_inf_n1 = tau_inf_n;
    };

    unsigned int n;
    vector<Tensor<2, dim>> tau_v_n, tau_v_n1;
    Tensor<2, dim> tau_inf_n, tau_inf_n1;
};

template<unsigned int dim>
class KronnerDecomp : public StateBase<dim> {
public:
    KronnerDecomp() : StateBase<dim>() {
        this->elastic_component = new ElasticState<dim>();
    };

    KronnerDecomp(KronnerDecomp<dim> *to_cpy)
            : StateBase<dim>(to_cpy), elastic_component(to_cpy->elastic_component) {};

    ElasticState<dim> *get_elastic_component() { return elastic_component; };

    virtual void update_elastic_component() = 0;

    void copy(KronnerDecomp<dim> *to_cpy) {
        StateBase<dim>::copy(to_cpy);
        this->elastic_component = to_cpy->elastic_component;
    };

//    TODO maybe make protected and force use of get_elastic_component()
    ElasticState<dim> *elastic_component;
protected:

};

template<unsigned int dim>
class RateIndependentPlasticityState : public KronnerDecomp<dim> {
public:
    RateIndependentPlasticityState()
            : KronnerDecomp<dim>(), Fp_n(Physics::Elasticity::StandardTensors<dim>::I),
              Fp_n1(Physics::Elasticity::StandardTensors<dim>::I), dlam_n(0), dlam_n1(0), ep_n(0), ep_n1(0), f_n(0),
              f_n1(0), plastic_flow_n(false), plastic_flow_n1(false) {};

    RateIndependentPlasticityState(RateIndependentPlasticityState<dim> *to_cpy)
            : KronnerDecomp<dim>(to_cpy), Fp_n(to_cpy->Fp_n), Fp_n1(to_cpy->Fp_n1), dlam_n(to_cpy->dlam_n),
              dlam_n1(to_cpy->dlam_n1), ep_n(to_cpy->ep_n), ep_n1(to_cpy->ep_n1), f_n(to_cpy->f_n), f_n1(to_cpy->f_n1),
              plastic_flow_n(to_cpy->plastic_flow_n), plastic_flow_n1(to_cpy->plastic_flow_n1) {};

    virtual StateBase<dim> *
    copy() override { return dynamic_cast<StateBase<dim> *>(new RateIndependentPlasticityState<dim>()); };

    void copy(RateIndependentPlasticityState<dim> *to_cpy) {
        KronnerDecomp<dim>::copy(to_cpy);
        this->Fp_n1 = to_cpy->Fp_n1;
        this->T_n1 = to_cpy->T_n1;
        this->dFp_n1 = to_cpy->dFp_n1;
        this->ep_n1 = to_cpy->ep_n1;
        this->dlam_n1 = to_cpy->dlam_n1;
        this->f_n1 = to_cpy->f_n1;
        this->plastic_flow_n1 = to_cpy->plastic_flow_n1;

        this->Fp_n = to_cpy->Fp_n;
        this->T_n = to_cpy->T_n;
        this->dFp_n = to_cpy->dFp_n;
        this->ep_n = to_cpy->ep_n;
        this->dlam_n = to_cpy->dlam_n;
        this->f_n = to_cpy->f_n;
        this->plastic_flow_n = to_cpy->plastic_flow_n;
    };

    Tensor<2, dim> Fp_n;
    Tensor<2, dim> Fp_n1;

    Tensor<2, dim> T_n;
    Tensor<2, dim> T_n1;

    Tensor<2, dim> dFp_n;
    Tensor<2, dim> dFp_n1;

    double dlam_n;
    double dlam_n1;

    double ep_n;
    double ep_n1;

    double f_n;
    double f_n1;

    bool plastic_flow_n;
    bool plastic_flow_n1;

    bool initial_nr_iteration = true;

    void update_elastic_component() override {
        this->elastic_component->F_n1 = this->F_n1 * invert(Fp_n1);
    };

    void update() override {
        update_elastic_component();
        this->elastic_component->update();
        StateBase<dim>::update();
        Fp_n = Fp_n1;
        T_n = T_n1;
        dFp_n = dFp_n1;
        ep_n = ep_n1;
        dlam_n = dlam_n1;
        f_n = f_n1;
        plastic_flow_n = plastic_flow_n1;
        initial_nr_iteration = true;
    };

    void reset() override {
        StateBase<dim>::reset();
        Fp_n1 = Fp_n;
        T_n1 = T_n;
        dFp_n1 = dFp_n;
        ep_n1 = ep_n;
        dlam_n1 = dlam_n;
        f_n1 = f_n;
        plastic_flow_n1 = plastic_flow_n;
    };

    void copy_new_values(RateIndependentPlasticityState<dim> *to_cpy) {
        Fp_n1 = to_cpy->Fp_n1;
        T_n1 = to_cpy->T_n1;
        dFp_n1 = to_cpy->dFp_n1;
        ep_n1 = to_cpy->ep_n1;
        dlam_n1 = to_cpy->dlam_n1;
        f_n1 = to_cpy->f_n1;
    };


    double scalar_output(ScalarOutputFlag flag) override {
        switch (flag) {
            case ScalarOutputFlag::EP:
                return ep_n1;
            case ScalarOutputFlag::DLAM:
                return dlam_n1;
            case ScalarOutputFlag::FPLAS:
                return f_n1;
            default:
                return StateBase<dim>::scalar_output(flag);
        }
    };

    Tensor<1, dim> vector_output(VectorOutputFlag flag) override {
        switch (flag) {
            default:
                return StateBase<dim>::vector_output(flag);
        }
    };

    Tensor<2, dim> tensor_output(TensorOutputFlag flag) override {
        switch (flag) {
            case TensorOutputFlag::T:
                return T_n1;
            case TensorOutputFlag::FP:
                return Fp_n1;
            case TensorOutputFlag::FE:
                return this->elastic_component->F_n1;
            default:
                return StateBase<dim>::tensor_output(flag);
        }
    };

};


template<unsigned int dim>
class ExplicitRateDependentPlasticityState : public KronnerDecomp<dim> {
public:
    ExplicitRateDependentPlasticityState()
            : KronnerDecomp<dim>(), Fp_n(Physics::Elasticity::StandardTensors<dim>::I),
              Fp_n1(Physics::Elasticity::StandardTensors<dim>::I), dlam_n(0), dlam_n1(0), ep_n(0), ep_n1(0), f_n(0),
              f_n1(0), plastic_flow_n(false), plastic_flow_n1(false) {};

    ExplicitRateDependentPlasticityState(ExplicitRateDependentPlasticityState<dim> *to_cpy)
            : KronnerDecomp<dim>(to_cpy), Fp_n(to_cpy->Fp_n), Fp_n1(to_cpy->Fp_n1), dlam_n(to_cpy->dlam_n),
              dlam_n1(to_cpy->dlam_n1), ep_n(to_cpy->ep_n), ep_n1(to_cpy->ep_n1), f_n(to_cpy->f_n), f_n1(to_cpy->f_n1),
              plastic_flow_n(to_cpy->plastic_flow_n), plastic_flow_n1(to_cpy->plastic_flow_n1) {};

    virtual StateBase<dim> *
    copy() override { return dynamic_cast<StateBase<dim> *>(new ExplicitRateDependentPlasticityState<dim>()); };

    void copy(ExplicitRateDependentPlasticityState<dim> *to_cpy) {
        KronnerDecomp<dim>::copy(to_cpy);
        this->Fp_n1 = to_cpy->Fp_n1;
        this->T_n1 = to_cpy->T_n1;
        this->dFp_n1 = to_cpy->dFp_n1;
        this->ep_n1 = to_cpy->ep_n1;
        this->dlam_n1 = to_cpy->dlam_n1;
        this->f_n1 = to_cpy->f_n1;
        this->plastic_flow_n1 = to_cpy->plastic_flow_n1;

        this->Fp_n = to_cpy->Fp_n;
        this->T_n = to_cpy->T_n;
        this->dFp_n = to_cpy->dFp_n;
        this->ep_n = to_cpy->ep_n;
        this->dlam_n = to_cpy->dlam_n;
        this->f_n = to_cpy->f_n;
        this->plastic_flow_n = to_cpy->plastic_flow_n;
    };

    Tensor<2, dim> Fp_n;
    Tensor<2, dim> Fp_n1;

    Tensor<2, dim> T_n;
    Tensor<2, dim> T_n1;

    Tensor<2, dim> dFp_n;
    Tensor<2, dim> dFp_n1;

    double dlam_n;
    double dlam_n1;

    double ep_n;
    double ep_n1;

    double f_n;
    double f_n1;

    bool plastic_flow_n;
    bool plastic_flow_n1;

    bool initial_nr_iteration = true;

    void update_elastic_component() override {
        this->elastic_component->F_n1 = this->F_n1 * invert(Fp_n1);
    };

    void update() override {
        update_elastic_component();
        this->elastic_component->update();
        StateBase<dim>::update();
        Fp_n = Fp_n1;
        T_n = T_n1;
        dFp_n = dFp_n1;
        ep_n = ep_n1;
        dlam_n = dlam_n1;
        f_n = f_n1;
        plastic_flow_n = plastic_flow_n1;
        initial_nr_iteration = true;
    };

    void reset() override {
        StateBase<dim>::reset();
        Fp_n1 = Fp_n;
        T_n1 = T_n;
        dFp_n1 = dFp_n;
        ep_n1 = ep_n;
        dlam_n1 = dlam_n;
        f_n1 = f_n;
        plastic_flow_n1 = plastic_flow_n;
    };

    void copy_new_values(ExplicitRateDependentPlasticityState<dim> *to_cpy) {
        Fp_n1 = to_cpy->Fp_n1;
        T_n1 = to_cpy->T_n1;
        dFp_n1 = to_cpy->dFp_n1;
        ep_n1 = to_cpy->ep_n1;
        dlam_n1 = to_cpy->dlam_n1;
        f_n1 = to_cpy->f_n1;
    };


    double scalar_output(ScalarOutputFlag flag) override {
        switch (flag) {
            case ScalarOutputFlag::ELASTIC_STRAIN_ENERGY:
                return this->elastic_component->elastic_strain_energy_n1;
            case ScalarOutputFlag::EP:
                return ep_n1;
            case ScalarOutputFlag::DLAM:
                return dlam_n1;
            case ScalarOutputFlag::FPLAS:
                return f_n1;
            default:
                return StateBase<dim>::scalar_output(flag);
        }
    };

    Tensor<1, dim> vector_output(VectorOutputFlag flag) override {
        switch (flag) {
            default:
                return StateBase<dim>::vector_output(flag);
        }
    };

    Tensor<2, dim> tensor_output(TensorOutputFlag flag) override {
        switch (flag) {
            case TensorOutputFlag::T:
                return T_n1;
            case TensorOutputFlag::FP:
                return Fp_n1;
            case TensorOutputFlag::FE:
                return this->elastic_component->F_n1;
            default:
                return StateBase<dim>::tensor_output(flag);
        }
    };

};

template<unsigned int dim>
class RateDependentPlasticityState : public RateIndependentPlasticityState<dim>{

    RateDependentPlasticityState() : RateIndependentPlasticityState<dim>() {};
    RateDependentPlasticityState(RateDependentPlasticityState<dim> *to_cpy)
    : RateIndependentPlasticityState<dim>(to_cpy) {};

    virtual StateBase<dim> *
    copy() override { return dynamic_cast<StateBase<dim> *>(new RateDependentPlasticityState<dim>()); };

};

template<unsigned int dim>
class IsoVolElasticState : public ElasticState<dim> {
public:
    double J_n;
    double J_n1;

    Tensor<2, dim> Fb_n;
    Tensor<2, dim> Fb_n1;

    double p_n;
    double p_n1;

    Tensor<2, dim> taub_n;
    Tensor<2, dim> taub_n1;

    SymmetricTensor<2, dim> cb_n;
    SymmetricTensor<2, dim> cb_n1;

    SymmetricTensor<4, dim> cv_n;
    SymmetricTensor<4, dim> cv_n1;
};




#endif //SOLVER_MATERIALSTATE_H
