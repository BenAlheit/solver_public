//
// Created by alhei on 2022/08/29.
//

#ifndef SOLVER_BOUNDARYCONDITION_H
#define SOLVER_BOUNDARYCONDITION_H

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <array>
#include <vector>

#include "../utils/utils.h"

using namespace std;
using namespace dealii;

template<unsigned int dim>
class BoundaryCondition {
public:
    explicit BoundaryCondition(unsigned int boundary_id) : boundary_id(boundary_id) {
        iota(dim_range.begin(), dim_range.end(), 0);
    };


    unsigned int get_boundary_id() const { return boundary_id; };

protected:
    const unsigned int boundary_id;
    array<unsigned int, dim> dim_range;
private:
};


template<unsigned int dim>
class DirichletBoundaryCondition : public BoundaryCondition<dim> {
public:
    DirichletBoundaryCondition() : BoundaryCondition<dim>(0), n_components(0) {};

    DirichletBoundaryCondition(unsigned int boundary_id,
                               unsigned int n_components)
            : BoundaryCondition<dim>(boundary_id), n_components(n_components) {};

    DirichletBoundaryCondition(unsigned int boundary_id,
                               const vector<unsigned int> &components,
                               const vector<double> &end_values,
                               const vector<double> &initial_values = {});

    virtual void apply_bc(const DoFHandler<dim> &dof_handler,
                          AffineConstraints<double> &constraints,
                          double inc_pct) const;

protected:
    vector<unsigned int> components;
    vector<double> end_values;
    vector<double> initial_values;
    vector<double> deltas;
    const unsigned int n_components;
    vector<unsigned int> comp_range;
    ComponentMask mask;

};

template<unsigned int dim>
DirichletBoundaryCondition<dim>::DirichletBoundaryCondition(unsigned int boundary_id,
                                                            const vector<unsigned int> &in_components,
                                                            const vector<double> &in_end_values,
                                                            const vector<double> &in_initial_values)
        : BoundaryCondition<dim>(boundary_id), components(in_components), end_values(in_end_values),
          n_components(components.size()) {
    comp_range = vector<unsigned int>(n_components);
    iota(comp_range.begin(), comp_range.end(), 0);

    mask = ComponentMask(dim, false);
    for (const auto &i_comp: components)
        mask.set(i_comp, true);

    if (in_initial_values.empty()) {
        initial_values = vector<double>(components.size(), 0.);
    } else {
        initial_values = in_initial_values;
    }

    deltas = vector<double>(dim);
    for (const auto &i: comp_range)
        deltas.at(components.at(i)) = end_values.at(i) - initial_values.at(i);
}

template<unsigned int dim>
void DirichletBoundaryCondition<dim>::apply_bc(const DoFHandler<dim> &dof_handler,
                                               AffineConstraints<double> &constraints,
                                               double inc_pct) const {
    vector<double> step_deltas(dim, 0);
    for (const auto &i: components)
        step_deltas.at(i) = deltas.at(i) * inc_pct;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             this->boundary_id,
                                             Functions::ConstantFunction<dim>(step_deltas),
                                             constraints,
                                             mask);
}


template<unsigned int dim>
class Slider : public DirichletBoundaryCondition<dim> {
public:
    Slider(unsigned int boundary_id,
           unsigned int normal_component,
           double initial_value = 0)
            : DirichletBoundaryCondition<dim>(boundary_id,
                                              vector<unsigned int>(1, normal_component),
                                              vector<double>(1, initial_value),
                                              vector<double>(1, initial_value)) {};

protected:

private:

};

template<unsigned int dim>
class Fix : public DirichletBoundaryCondition<dim> {
public:
    explicit Fix(unsigned int boundary_id,
                 const vector<double> &initial_values = vector<double>(dim, 0))
            : DirichletBoundaryCondition<dim>(boundary_id,
                                              vector<unsigned int>({0, 1, 2}),
                                              initial_values,
                                              initial_values) {};

protected:


private:

};

template<int dim>
class PBCNode {

public:
    PBCNode() {};

    PBCNode(Point<dim> position,
            vector<unsigned int> global_dofs) :
            position(position),
            global_dofs(global_dofs) {};

    Point<dim> position;
    vector<unsigned int> global_dofs;


    friend bool operator==(const PBCNode &n1, const PBCNode &n2) {
        return n1.global_dofs.at(0) == n2.global_dofs.at(0);
    };

};

template<unsigned int dim>
class PeriodicBoundaryCondition : public DirichletBoundaryCondition<dim> {
public:
    typedef pair<unsigned int, unsigned int> face_pair;
    typedef pair<PBCNode<dim>, PBCNode<dim>> node_pair;

    PeriodicBoundaryCondition(const DoFHandler<dim> &dof_handler,
                              const Triangulation<dim> &triangulation,
                              const vector<face_pair> &boundary_pairs,
                              const Tensor<2, dim> & Grad_u_end,
                              const Tensor<2, dim> & Grad_u_start = Tensor<2, dim>());

    void apply_bc(const DoFHandler<dim> &dof_handler,
                  AffineConstraints<double> &constraints,
                  double inc_pct) const override;

private:

    Tensor<2, dim> Delta_u;
    Tensor<2, dim> Grad_u_start;
    Tensor<2, dim> Grad_u_end;
    array<vector<node_pair>, dim> node_pairs;
    PBCNode<dim> anchor_xyz, anchor_xy, anchor_x;

};

template<unsigned int dim>
PeriodicBoundaryCondition<dim>::PeriodicBoundaryCondition(const DoFHandler<dim> &dof_handler,
                                                          const Triangulation<dim> &triangulation,
                                                          const vector<face_pair> &boundary_pairs,
                                                          const Tensor<2, dim> & Grad_u_end,
                                                          const Tensor<2, dim> & Grad_u_start)
        : Grad_u_start(Grad_u_start), Grad_u_end(Grad_u_end) {
    array<vector<face_pair>, dim> face_pair_indices, dof_pair_indices;
//    array<vector<unsigned int>, dim> neg_faces, pos_faces;
    array<vector<PBCNode<dim>>, dim> neg_nodes, pos_nodes;
//    array<vector<node_pair>, dim> node_pairs;
    Point<dim> p1, p2;
//    vector<bool> pos_p_found(dim, false), neg_p_found(dim, false);

    vector<unsigned int> p_ids_found;
    unsigned int counter = 0;

    typedef vector<typename Triangulation<dim>::vertex_iterator> v_iterator_list;
    map<unsigned int, v_iterator_list> used_verts;
    for (const auto &b_pair: boundary_pairs) {
        used_verts[b_pair.first] = v_iterator_list();
        used_verts[b_pair.second] = v_iterator_list();
    }

    vector<unsigned int> v_dofs(dim);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        if (cell->at_boundary()) {
            for (auto const &face: cell->face_iterators()) {
                if (face->at_boundary()) {
                    if (p_ids_found.size() < dim * 2) {
                        if (find(p_ids_found.begin(), p_ids_found.end(), face->boundary_id()) == p_ids_found.end()) {
                            for (const auto &p: boundary_pairs) {
                                if (p.first == face->boundary_id()) {
                                    p1[counter] = face->center()[counter];
                                    break;
                                }
                                if (p.second == face->boundary_id()) {
                                    p2[counter] = face->center()[counter];
                                    break;
                                }
                                counter++;
                            }
                            counter = 0;
                            p_ids_found.push_back(face->boundary_id());
                        }
                    }
                    for (const auto &index: face->vertex_indices()) {
                        auto vertex = face->vertex(index);
                        if (find(used_verts.at(face->boundary_id()).begin(),
                                 used_verts.at(face->boundary_id()).end(),
                                 face->vertex_iterator(index)) == used_verts.at(face->boundary_id()).end()) {
                            for (const auto &comp: this->dim_range)
                                v_dofs.at(comp) = face->vertex_dof_index(index, comp);
                            for (const auto &comp: this->dim_range) {
                                if (boundary_pairs.at(comp).first == face->boundary_id()) {
                                    neg_nodes.at(comp).push_back(PBCNode<dim>(vertex, v_dofs));
//                                    break;
                                }
                                if (boundary_pairs.at(comp).second == face->boundary_id()) {
                                    pos_nodes.at(comp).push_back(PBCNode<dim>(vertex, v_dofs));
//                                    break;
                                }
                                counter++;
                            }
                            counter = 0;
                            used_verts.at(face->boundary_id()).push_back(face->vertex_iterator(index));
                        }
                    }
                }
            }
        }
    }

    Tensor<1, dim> dist = p2 - p1;
    Point<dim> projected_neg, anchor_xyz_pos, anchor_xy_pos, anchor_x_pos;

    anchor_xyz_pos = p1;
    anchor_xy_pos = p1;
    anchor_x_pos = p1;

    anchor_xy_pos[2] += dist[2];
    anchor_x_pos[1] += dist[1];

    counter = 0;
    bool find_neighbour = true;
    for (const auto &comp: this->dim_range) {
        for (const auto &neg_node: neg_nodes.at(comp)) {

//            for (unsigned int i_check = 0; i_check < comp; ++i_check) {
//                for (const auto &neg_node_check: neg_nodes.at(i_check)) {
//                    if (neg_node == neg_node_check) {
//                        find_neighbour = false;
//                        break;
//                    }
//                }
//                if(find_neighbour){
//                    for (const auto &pos_node_check: pos_nodes.at(i_check)) {
//                        if (neg_node == pos_node_check) {
//                            find_neighbour = false;
//                            break;
//                        }
//                    }
//                }
//                if (not find_neighbour)
//                    break;
//            }

            if (almost_equals((neg_node.position - anchor_xyz_pos).norm(), 0))
                anchor_xyz = neg_node;
            if (almost_equals((neg_node.position - anchor_xy_pos).norm(), 0))
                anchor_xy = neg_node;
            if (almost_equals((neg_node.position - anchor_x_pos).norm(), 0))
                anchor_x = neg_node;

            if (find_neighbour) {
                projected_neg = neg_node.position;
                projected_neg[comp] += dist[comp];

                for (const auto &pos_node: pos_nodes.at(comp)) {

                    if (almost_equals((pos_node.position - projected_neg).norm(), 0)) {
                        node_pairs.at(comp).push_back(node_pair(neg_node, pos_node));
                        break;
                    }
                    counter++;
                }
//                pos_nodes.at(comp).erase(pos_nodes.at(comp).begin() + counter);
                counter = 0;
            }
            find_neighbour = true;
        }
    }

    Delta_u = Grad_u_end - Grad_u_start;
    for (const auto &i: this->dim_range) {
        for (const auto &j: this->dim_range) {
            Delta_u[i][j] *= dist[j];
        }
    }

}

template<unsigned int dim>
void PeriodicBoundaryCondition<dim>::apply_bc(const DoFHandler<dim> &dof_handler,
                                              AffineConstraints<double> &constraints,
                                              double inc_pct) const {
    for (const auto &j: this->dim_range) {
        constraints.add_line(anchor_xyz.global_dofs.at(j));

        constraints.add_line(anchor_xy.global_dofs.at(j));
        constraints.set_inhomogeneity(anchor_xy.global_dofs.at(j), inc_pct * Delta_u[2][j]);

        constraints.add_line(anchor_x.global_dofs.at(j));
        constraints.set_inhomogeneity(anchor_x.global_dofs.at(j), inc_pct * Delta_u[1][j]);
    }

    for (const auto &i: this->dim_range) {
        for(const auto & p : node_pairs.at(i))
        for (const auto &j: this->dim_range) {
            if(not constraints.is_constrained(p.second.global_dofs.at(j))){
                constraints.add_line(p.second.global_dofs.at(j));
                constraints.add_entry(p.second.global_dofs.at(j), p.first.global_dofs.at(j), 1);
                constraints.set_inhomogeneity(p.second.global_dofs.at(j), inc_pct * Delta_u[i][j]);
            }
        }
    }
}

template<unsigned int dim>
class NeumannBoundaryCondition : BoundaryCondition<dim> {
public:

protected:

private:
};

#endif //SOLVER_BOUNDARYCONDITION_H
