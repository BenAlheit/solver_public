#include <iostream>
#include "Solver.h"
#include "SolverExample.h"


#include "examples/CubeWithIMP.h"


#include "examples/Cube.h"
#include "examples/ElastoPlasticCube.h"
#include "examples/QuarterCylinder.h"
#include "examples/NeckingCylinder.h"
#include "examples/PSC.h"

#include "examples/check-approximate-tangent/CheckApproximateTangentElastic.h"
#include "examples/check-approximate-tangent/CheckApproximateTangentViscoelastic.h"

#include "examples/cyclical-tension-compression/ElasticCyclicalTensionCompression.h"
#include "examples/cyclical-tension-compression/ViscoCyclicalTensionCompression.h"
#include "examples/cyclical-tension-compression/ExplicitRateDependentPlasticityCyclicalTensionCompression.h"
#include "examples/cyclical-tension-compression/ExplicitRateDependentCrystalPlasticityCyclicalTensionCompression.h"

#include "examples/cube-with-sphere/CubeWithSphere.h"
#include "examples/cube-with-sphere/CubeWithSpherePBC.h"
#include "examples/cube-with-sphere/ViscoplasticCubeWithSpherePBC.h"
#include "examples/cube-with-sphere/ViscoCrystalplasticCubeWithSpherePBC.h"
#include "examples/cube-with-sphere/FCCViscoCrystalplasticCubeWithSpherePBC.h"

#include "examples/polycrystal/PolycrystalPBC.h"
#include "examples/polycrystal-imp/PolycrystalIMPPBC.h"

// TODO Documentation
// TODO Documentation
// TODO Documentation
// TODO Documentation
// TODO Documentation

// TODO Parallelize output (ignore ghosting for now)
// TODO Make parallelization better (ghost dofs, add periodicity)
// TODO Use VectorTools for projection (actually maybe not)

int main(int argc, char **argv){
//    MultithreadInfo::set_thread_limit(1);

    using namespace dealii;
    using namespace SolverExample;
    const unsigned int dim = 3;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

//    PSC<dim> problem = PSC<dim>();


//    auto problem = CubeWithIMP<dim>();

//    auto problem = PolycrystalPBC<dim>();
//    auto problem = PolycrystalIMPPBC<dim>();

//    auto problem = CheckApproximateTangentElastic<dim>();
//    auto problem = CheckApproximateTangentViscoelastic<dim>();

//    auto problem = ElastoPlasticCube<dim>();
//    auto problem = QuarterCylinder<dim>();
//    auto problem = NeckingCylinder<dim>();


//    auto problem = ElasticCyclicalTensionCompression<dim>();
//    auto problem = ViscoCyclicalTensionCompression<dim>();
//    auto problem = ExplicitRateDependentPlasticityCyclicalTensionCompression<dim>();
//    auto problem = ExplicitRateDependentCrystalPlasticityCyclicalTensionCompression<dim>();

//    auto problem = Cube<dim>();
//    auto problem = CubeWithSphere<dim>();
//    auto problem = CubeWithSpherePBC<dim>();
//    auto problem = ViscoplasticCubeWithSpherePBC<dim>();
//    auto problem = ViscoCrystalplasticCubeWithSpherePBC<dim>();
    auto problem = FCCViscoCrystalplasticCubeWithSpherePBC<dim>();


//    ElasticProblem<2> elastic_problem;
//    elastic_problem.run();

    return 0;
}
