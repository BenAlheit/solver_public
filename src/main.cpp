#include <iostream>
#include "Solver.h"
#include "SolverExample.h"
#include "examples/CubeWithIMP.h"
#include "examples/CubeWithSphere.h"
#include "examples/CubeWithSpherePBC.h"
#include "examples/Cube.h"
#include "examples/ElastoPlasticCube.h"
#include "examples/QuarterCylinder.h"
#include "examples/NeckingCylinder.h"
#include "examples/CheckApproximateTangent.h"
#include "examples/PSC.h"

#include "examples/polycrystal/PolycrystalPBC.h"
#include "examples/polycrystal-imp/PolycrystalIMPPBC.h"

#include "examples/cyclical-tension-compression/ViscoCyclicalTensionCompression.h"
#include "examples/cyclical-tension-compression/ElasticCyclicalTensionCompression.h"

// TODO Documentation
// TODO Documentation
// TODO Documentation
// TODO Documentation
// TODO Documentation

// TODO Viscoplastic material
// TODO Viscocrystalplastic material
// TODO Fix parallelization

int main(int argc, char **argv){
//    MultithreadInfo::set_thread_limit(1);

    using namespace dealii;
    using namespace SolverExample;
    const unsigned int dim = 3;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

//    PSC<dim> problem = PSC<dim>();


//    Cube<dim> problem = Cube<dim>();
//    CubeWithSphere<dim> problem = CubeWithSphere<dim>();
//    CubeWithIMP<dim> problem = CubeWithIMP<dim>();

//    CubeWithSpherePBC<dim> problem = CubeWithSpherePBC<dim>();
//    PolycrystalPBC<dim> problem = PolycrystalPBC<dim>();
//    PolycrystalIMPPBC<dim> problem = PolycrystalIMPPBC<dim>();


//    CheckApproximateTangent<dim> problem = CheckApproximateTangent<dim>();
//    ElastoPlasticCube<dim> problem = ElastoPlasticCube<dim>();
//    QuarterCylinder<dim> problem = QuarterCylinder<dim>();
//    NeckingCylinder<dim> problem = NeckingCylinder<dim>();


//    ElasticCyclicalTensionCompression<dim> problem = ElasticCyclicalTensionCompression<dim>();
    ViscoCyclicalTensionCompression<dim> problem = ViscoCyclicalTensionCompression<dim>();


    //    ElasticProblem<2> elastic_problem;
//    elastic_problem.run();

    return 0;
}
