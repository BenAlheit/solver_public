//
// Created by alhei on 2022/09/07.
//

#ifndef SOLVER_PSC_H
#define SOLVER_PSC_H


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include "../materials/OutputFlags.h"
#include "../problem/FEMesh.h"
#include "../Solver.h"

using namespace std;
using namespace dealii;

template<unsigned int dim>
class PSC {
public:
    explicit PSC(unsigned int n_refinements = 0);

    Triangulation<dim> triangulation;
    map<unsigned int, Material<dim> *> materials;
    vector<Stage<dim>> stages;
    Time time;

private:
    array<unsigned int, dim> range;
    const string name = "PSC-slip";

    Point<dim> p1 = Point<dim>({0, 0, 0});
//    Point<dim> p2 = Point<dim>({10, 7.5, 15});
//    vector<unsigned int> repetitions = vector<unsigned int>({4, 3, 6});
    Point<dim> p2 = Point<dim>({10, 7.5, 2.5});
    vector<unsigned int> repetitions = vector<unsigned int>({4, 3, 1});

    const unsigned int order = 1;

    void make_mesh();

    const unsigned int n_refinements;
    const unsigned int n_refine_close = 3;
    const double refine_close_domain = 0.3;
    const double push = 5;



//    Point<dim> p1 = Point<dim>({0, 0, 0});
//    Point<dim> p2 = Point<dim>({1, 1, 1});

};

template<unsigned int dim>
PSC<dim>::PSC(unsigned int n_refinements): n_refinements(n_refinements) {
    iota(range.begin(), range.end(), 0);

    make_mesh();
    const double kappa = 44.12e3;
    const double mu = 16.92e3;
    materials[/*material id*/ 0] = new NeoHookean<dim>(/*material id*/ 0, kappa, mu);

    unsigned int n_steps = 50;
    unsigned int n_out = 25;

    time = Time(/*end_time*/ 1,
                             n_steps,
                             n_out);

    vector<DirichletBoundaryCondition<dim> *> dbcs;
    for (const auto &i: range)
        dbcs.push_back(new Slider<dim>(2 + i, i));

    dbcs.push_back(new Slider<dim>(7, 2));

//    dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 1,
//            /*components*/ vector<unsigned int>({0, 1, 2}),
//            /*magnitudes*/ vector<double>({0, -push, 0})));
//    dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 1,
//            /*components*/ vector<unsigned int>({1}),
//            /*magnitudes*/ vector<double>({-push})));
    dbcs.push_back(new DirichletBoundaryCondition<dim>( /*boundary_id*/ 1,
            /*components*/ vector<unsigned int>({1, 2}),
            /*magnitudes*/ vector<double>({-push, 0})));

    vector<unsigned int> material_ids({0});

    map<OutputFlags::ScalarOutputFlag, vector<unsigned int>> scalar_outputs;
    scalar_outputs[OutputFlags::ScalarOutputFlag::J] = material_ids;
    scalar_outputs[OutputFlags::ScalarOutputFlag::P] = material_ids;

    map<OutputFlags::VectorOutputFlag, vector<unsigned int>> vector_outputs;
    vector_outputs[OutputFlags::VectorOutputFlag::NB1] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB2] = material_ids;
    vector_outputs[OutputFlags::VectorOutputFlag::NB3] = material_ids;

    map<OutputFlags::TensorOutputFlag, vector<unsigned int>> tensor_outputs;
    tensor_outputs[OutputFlags::TensorOutputFlag::F] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRAIN] = material_ids;
    tensor_outputs[OutputFlags::TensorOutputFlag::STRESS] = material_ids;

    vector<OutputFlags::MeshOutputFlag> mesh_outputs({OutputFlags::MeshOutputFlag::MATERIAL_ID,
                                                      OutputFlags::MeshOutputFlag::BOUNDARY_ID});

    Stage<dim> first_stage(&time,
                           dbcs,
                           vector<NeumannBoundaryCondition<dim> *>({}),
                           scalar_outputs,
                           vector_outputs,
                           tensor_outputs,
                           mesh_outputs);

    stages.push_back(first_stage);
    FEMesh<dim> mesh(order,
                     materials,
                     triangulation);

    Problem<dim> problem(name, &mesh, stages, &time);
    Solver::Solver<dim> solver(&problem);
    solver.solve();

}

template<unsigned int dim>
void PSC<dim>::make_mesh() {
    GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

    for (auto &face: triangulation.active_face_iterators()) {
        if (face->at_boundary()) {
            if (almost_equals(face->center()[1], p2[1]) &&  face->center()[0] < p2[0] * 0.5) {
                face->set_boundary_id(1);
            } else {
                for (const auto &i: range) {
                    if (almost_equals(face->center()[i], p1[i])) {
                        face->set_boundary_id(i + 2);
                        break;
                    } else if (almost_equals(face->center()[i], p2[i])) {
                        face->set_boundary_id(i + 5);
                        break;
                    }
                }
            }
        }
    }

    if (n_refine_close > 0) {
        Point<dim> mesh_point;
        for (int i = 0; i < n_refine_close; ++i) {
            for (auto &cell: triangulation.active_cell_iterators()) {
                if (cell->at_boundary()) {
                    for (unsigned int i_vert = 0; i_vert < cell->n_vertices(); i_vert++) {
                        mesh_point = cell->vertex(i_vert);
                        if ((p2[0] * refine_close_domain <= mesh_point[0] && mesh_point[0] <= p2[0] * 0.5)
                            && almost_equals(mesh_point[1], p2[1])) {
                            cell->set_refine_flag();
                            break;
                        }
                    }
                }
            }
            triangulation.execute_coarsening_and_refinement();
        }
    }

    if (n_refinements > 0)
        triangulation.refine_global(n_refinements);

}



#endif //SOLVER_PSC_H
