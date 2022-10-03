# Solver

This project is currently in very early stages of its development.
The intention is to develop a general purpose finite element code 
to model a large variety of computational mechanics problems. Currently, only
solid mechanics problems are considered. However, the intention is to
develop functionality for multi-physics problems.

## Current features
* Parallelization for distributed memory machines.
* Finite strain solid mechanics solver.
* Support for heterogeneous structures. 
* Various sophisticated finite strain material models:
  - Rate-dependent crystal plasticity.
  - Rate-dependent plasticity.
  - Viscoelasticity.
  - Hyperelasticity.
  - Anisotropy.
* Periodic boundary conditions.
* Homogenization operations.

## Current development direction
* Implementation of Geometric-MultiGrid (GMG) preconditioning.
* Implementation of matrix-free solver with GMG preconditioning.
* Implementation of Arbitrary Lagrangian-Eulerian (ALE) for plasticity (see motivation for this in example below).
* Implementation of adaptive remeshing for materials with history variables (see motivation for this in example below).

## Example of functionality (a toy problem): modelling a polycrystalline material with a elastic intermetallic particles (IMPs) 
### Problem setting

During the solidification of various metal alloys non-metallic elements precipitate out of the solution. This results in intermetallic particles that are embedded in the typical polycrystalline metal microstructure as illustrated below (see [this repo](https://github.com/BenAlheit/vtk-collection-mesher_public) for the code that generated the mesh on the left-hand side and [this repo](https://github.com/BenAlheit/imp-image-analysis_public) for the code that extracted particle geometries from optical micrographs).

![alt text](https://github.com/BenAlheit/solver_public/blob/master/imgs/setting.png?raw=true)

Note that a cross-section of the representative volume element (RVE) is presented above so that the intermetallic particles are visible. The unlabelled coloured portions of the mesh are different crystal grains. Some of the mathematical components of the model are loosely described in what follows. For the sake of brevity, it is assumed that the reader has some knowledge of finite strain continuum mechanics and the accompanying standard notation.

One seeks a displacement field as a function of position $x$ and time $t$, $u\left(x,\\,t\right)$, that satisfies the equilibrium equation, that is,

$\text{div}\\,\sigma=0 \qquad \text{for } x\in\Omega,\\,t\in[0,\\,T],$

and periodic boundary conditions, that is,

$u^+ - u^- = \left[F_M-I\right]\left[X^+ - X^-\right] \qquad \text{for } x\in\partial\Omega.$

Here, $\sigma$ denotes the Cauchy stress, $\Omega$ denotes the domain of the problem, $\partial\Omega$ denotes the boundary, $F_M$ denotes a macroscale deformation gradient $X$ denotes the initial position of a material point, and $\bullet^+$ and $\bullet^-$ denote pairwise values on opposite sides of the boundary.

Macroscale plane strain loading is applied and macroscale incompressibility is assumed, leading to the following expression:

$F_M-I = \text{diag}\left[ \tfrac{-\epsilon}{1+\epsilon} \qquad \epsilon \qquad 0\right]t,$

where $\epsilon$ is strain-like input parameter, similar to that controlled in a physical plane strain compression test.

The domain is made up of several subdomains describing intermetallic particles and separate crystal grains; that is,

$\Omega = \Omega_1 \cup \Omega_2 \cup ... \cup \Omega_n.$

Constitutive equations (denoted by $\hat{\bullet}$) for the stress and evolution of history variables $\xi$ are defined on each subdomain; that is,

$\sigma = \hat{\sigma}_i\left(\text{Grad}\\,u,\\, \xi\right) \qquad \text{on} \qquad \Omega_i,$

$\dot{\xi} = \hat{\dot{\xi}}_i\left(\text{Grad}\\,u,\\, \xi\right) \qquad \text{on} \qquad \Omega_i.$

The stress in subdomains relating to IMPs is given by

$\hat{\sigma}_{IMP}\left(F\right) = J^{-1}\mu\left[\bar{B}-\tfrac{1}{3}\text{tr}(\bar{B})I\right]+ \tfrac{1}{2} \kappa \left[J-J^{-1}\right]I,$ 

where $I$ is the identity tensor, $F=I+\text{Grad}\\,u$ is the deformation gradient, $J=\text{det}\\,F$ is the Jacobian, $\bar{B}=J^{-2/3}FF^T$ is the isochoric left Cauchy-Green tensor, $\mu$ is the shear modulus, and $\kappa$ is the bulk modulus. The IMPs are modelled as elastic; that is, loading history is not considered. The crystal grains are modelled using rate-dependent crystal plasticity. A Kroner decomposition of the deformation gradient is taken; that is,

$F=F^e F^p,$

where $F^e$ and $F^p$ are the elastic and plastic parts of the deformation gradient, respectively. The evolution of $F^p$ is governed by

$\dot{F^p} = \left[ \displaystyle \sum_{\alpha}^N  \nu^\alpha s^\alpha \otimes m^\alpha \right] F^p,$

where $N$ is the number of "slip systems" for the given crystal, $\bullet^\alpha$ denotes a value for a particular system, $\nu$ is the slip rate, $s$ is the slip direction and $m$ is the slip plane normal (orthogonal to $s$). The slip rate is given by

$\nu ^ \alpha = \tfrac{1}{\mu_{p}}\left[\frac{\tau^\alpha}{\sigma_y}\right]^{1/m} \text{sgn}\tau^\alpha,$

where $\mu_{p}$ is the viscoplastic viscocity, $\sigma_y$ is a yield stress-like parameter, $m$ is a material parameter, $\text{sgn}\bullet$ denotes the sign of $\bullet$, and $\tau^{\alpha}$ is the resolved Schmidt stress for system $\alpha$, given by

$\tau^{\alpha} = J^{-1}\left[F^e s^\alpha\right] \cdot \sigma \left[\left[ F^e\right]^{-T} m^\alpha\right].$

Note, that this is a simplified version of rate-dependent crystal plasticity analogous to perfect plasticity - in a more general case, $\sigma_y$ may be dependent on additional history/state variables. The stress is then given by the same equation as for the IMPs except the elastic deformation gradient is the argument; that is,

$\hat{\sigma}\_{CP}\left(F\right) = \hat{\sigma}_{IMP}\left(F^e\right)$. 

Each crystal is modelled as a face centred cubic (FCC) crystal with 12 independent slip systems. The crystal orientation is randomly assigned in the initial state from one crystal to the next, but constant within a single crystal.

The numerical treatment of the above is ommitted for brevity.

### Some results

The elastic strain energy density field, given by 

$\Psi_{IMP}\left(F\right) = \tfrac{1}{2} \mu\left[\text{tr}\left(\bar{B}\right)-3\right] + \tfrac{1}{2} \kappa \left[\frac{J^2 -1}{2}-\text{ln}J\right]$

for the IMPs and 

$\Psi^e\_{CP}\left(F\right) = \Psi_{IMP}\left(F^e\right) $

for the crystal grains, is presented for three different times in the simulation below.

![alt text](https://github.com/BenAlheit/solver_public/blob/master/imgs/se.png?raw=true)

Clearly, the strain energy density is larger in the IMPs. The volume averaged first Piol-Kirchhoff stress is presented below for varying degrees of mesh refinement and for microstructures that contain IMPs as well as microstructures that do not. 

![alt text](https://github.com/BenAlheit/solver_public/blob/master/imgs/fcc-polycrystal-first-stresses.png?raw=true)

The top centre graph in the figure above is analogous to the results one might obtain from a physical plane strain compression test.

It is apparent that, in this toy problem at least, the IMPs have an insignificant effect on the stress response.
