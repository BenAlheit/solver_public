//
// Created by alhei on 2022/08/31.
//

#ifndef SOLVER_OUTPUTFLAGS_H
#define SOLVER_OUTPUTFLAGS_H

#include "../Exceptions.h"
#include <string>

using namespace std;


namespace OutputFlags {
    enum class ScalarOutputFlag {
        P, J, EP, DLAM, FPLAS, ELASTIC_STRAIN_ENERGY
    };

    enum class VectorOutputFlag {
        NS1, NS2, NS3, NB1, NB2, NB3
    };

    enum class TensorOutputFlag {
        F, STRESS, STRAIN, T, FP, FE, FirstPiolaStress
    };

    enum class MeshOutputFlag {
        MATERIAL_ID, BOUNDARY_ID, ELEMENT_QUALITY
    };

    string to_string(const MeshOutputFlag &flag) {
        switch (flag) {
            case MeshOutputFlag::MATERIAL_ID:
                return "Material_ID";
            case MeshOutputFlag::BOUNDARY_ID:
                return "Boundary_ID";
            case MeshOutputFlag::ELEMENT_QUALITY:
                return "Element_quality";
            default:
                throw NotImplemented("to_string function not implemented for this MeshOutputFlag");
        }
    }

    string to_string(const ScalarOutputFlag &flag) {
        switch (flag) {
            case ScalarOutputFlag::J:
                return "Jacobian";
            case ScalarOutputFlag::P:
                return "Pressure";
            case ScalarOutputFlag::EP:
                return "Equivalent_plastic_strain";
            case ScalarOutputFlag::DLAM:
                return "Plastic_strain_rate";
            case ScalarOutputFlag::FPLAS:
                return "Plastic_yield_surface";
            case ScalarOutputFlag::ELASTIC_STRAIN_ENERGY:
                return "Elastic_strain_energy";
            default:
                throw NotImplemented("to_string function not implemented for this ScalarOutputFlag");
        }
    }

    template<unsigned int dim>
    vector<string> create_string_vector_for_vector(const string &name) {
        vector<string> out;
        array<unsigned int, dim> range;
        iota(range.begin(), range.end(), 0);
        for (const auto i: range)
            out.push_back(name + std::to_string(i + 1));
        return out;
    }

    template<unsigned int dim>
    vector<string> to_string(const VectorOutputFlag &flag) {
        switch (flag) {
            case VectorOutputFlag::NS1:
                return create_string_vector_for_vector<dim>("PrincipleStress1");
            case VectorOutputFlag::NS2:
                return create_string_vector_for_vector<dim>("PrincipleStress2");
            case VectorOutputFlag::NS3:
                return create_string_vector_for_vector<dim>("PrincipleStress3");
            case VectorOutputFlag::NB1:
                return create_string_vector_for_vector<dim>("PrincipleStretch1");
            case VectorOutputFlag::NB2:
                return create_string_vector_for_vector<dim>("PrincipleStretch2");
            case VectorOutputFlag::NB3:
                return create_string_vector_for_vector<dim>("PrincipleStretch3");
            default:
                throw NotImplemented("to_string function not implemented for this VectorOutputFlag");
        }
    }

    template<unsigned int dim>
    vector<string> create_string_vector_for_tensor(const string &name) {
        vector<string> out;
        array<unsigned int, dim> range;
        iota(range.begin(), range.end(), 0);
        for (const auto i: range)
            for (const auto j: range)
                out.push_back(name + std::to_string(i + 1) + std::to_string(j + 1));
        return out;
    }

    template<unsigned int dim>
    vector<string> to_string(const TensorOutputFlag &flag) {
        switch (flag) {
            case TensorOutputFlag::F:
                return create_string_vector_for_tensor<dim>("DeformationGradient");
            case TensorOutputFlag::FE:
                return create_string_vector_for_tensor<dim>("ElasticDeformationGradient");
            case TensorOutputFlag::FP:
                return create_string_vector_for_tensor<dim>("PlasticDeformationGradient");
            case TensorOutputFlag::STRESS:
                return create_string_vector_for_tensor<dim>("Stress");
            case TensorOutputFlag::FirstPiolaStress:
                return create_string_vector_for_tensor<dim>("FirstPiolaStress");
            case TensorOutputFlag::STRAIN:
                return create_string_vector_for_tensor<dim>("Strain");
            default:
                throw NotImplemented("to_string function not implemented for this TensorOutputFlag");
        }
    }

    enum class nScalarOutputFlag {
        H, ta, nu
    };

    string to_string(const nScalarOutputFlag &flag, const unsigned int &i) {
        switch (flag) {
            case nScalarOutputFlag::H:
                return "History_variable_" + std::to_string(i);
            default:
                throw NotImplemented("to_string function not implemented for this ScalarOutputFlag");
        }
    }

    struct nScalarOutput {
        nScalarOutput(const unsigned int &n,
                      const nScalarOutputFlag &flag)
                : n(n), flag(flag) {};

        const unsigned int n;
        const nScalarOutputFlag flag;

    };

    struct nScalarOutputHash {
        size_t operator()(const nScalarOutput &other) const {
            return hash<unsigned int>()(other.n) ^ hash<nScalarOutputFlag>()(other.flag);
        }
    };

    enum class nVectorOutputFlag {
        M, S
    };

    template<unsigned int dim>
    string to_string(const nVectorOutputFlag &flag, const unsigned int &i) {
        switch (flag) {
            case nVectorOutputFlag::M:
                return create_string_vector_for_vector<dim>("System_normal_" + std::to_string(i));
            case nVectorOutputFlag::S:
                return create_string_vector_for_vector<dim>("System_slip_" + std::to_string(i));
            default:
                throw NotImplemented("to_string function not implemented for this nVectorOutputFlag");
        }
    }

    struct nVectorOutput {
        nVectorOutput(const unsigned int &n,
                      const nVectorOutputFlag &flag)
                : n(n), flag(flag) {};

        const unsigned int n;
        const nVectorOutputFlag flag;
    };

    struct nVectorOutputHash {
        size_t operator()(const nVectorOutput &other) const {
            return hash<unsigned int>()(other.n) ^ hash<nVectorOutputFlag>()(other.flag);
        }
    };

    enum class nTensorOutputFlag {
        SYS
    };

    template<unsigned int dim>
    string to_string(const nTensorOutputFlag &flag, const unsigned int &i) {
        switch (flag) {
            case nTensorOutputFlag::SYS:
                return create_string_vector_for_tensor<dim>("System_" + std::to_string(i));
            default:
                throw NotImplemented("to_string function not implemented for this nTensorOutputFlag");
        }
    }

    struct nTensorOutput {
        nTensorOutput(const unsigned int &n,
                      const nTensorOutputFlag &flag)
                : n(n), flag(flag) {};

        const unsigned int n;
        const nTensorOutputFlag flag;
    };

    struct nTensorOutputHash {
        size_t operator()(const nTensorOutput &other) const {
            return hash<unsigned int>()(other.n) ^ hash<nTensorOutputFlag>()(other.flag);
        }
    };
}


#endif //SOLVER_OUTPUTFLAGS_H
