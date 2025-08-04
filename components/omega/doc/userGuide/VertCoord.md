(omega-user-vert-coord)=

## Vertical Coordinate

### Overview

Omega uses pseudo height, $\tilde{z} = \frac{p}{\rho_0 g}$,  as the vertical coordinate ([V0 governing equation document](OmegaV1GoverningEqns)).
The pseudo height is essentially a normalized pressure coordinate, with the advantage that it has units of meters.
The `VerticalCoord` class contains variables and functions relevant to keeping track of:
 - the bottom depth of each cell (read in from the mesh file)
 - the location of active vertical layers (used to set extents of vertical loop bounds):
   - the number of vertical layers in each cell (read in from the mesh file)
   - the max and min number of layers on edges and vertices at the bottom and top of the water column (computed from the number of cell levels)
 - pressure (computed from the layer thickness and surface pressure)
 - $z$ height (computed from the bottom depth, specific volume, and layer thickness)
 - geopotential (computed from the z height and tidal forcing)
 - desired vertical interface locations (computed from pressure, reference layer pseudo thickness, and user-specified weights)

Multiple instances of the vertical coordinate class can be created and accessed by a unique name.

### Variables

| Variable Name | Description | Units |
| ------------- | ----------- | ----- |
| NVertLevels   | maximum number of vertical layers | - |
| NVertLevelsP1 | maximum number of vertical layers plus 1 | - |
| PressureInterface | pressure at layer interfaces | pressure per unit area at layer interfaces | kg m/s$^2$ |
| PressureMid | pressure at layer mid points | pressure per unit area at layer mid point | kg m/s$^2$ |
| ZInterface | z height of layer interfaces | m |
| ZMid | z height of layer midpoint | m |
| GeopotentialMid | geopotential at layer mid points | m$^2$/s$^2$|
| LayerThicknessPStar | desired layer thickness based on total perturbation from the reference thickness | - |
| MinLevelCell | first active layer for cell | - |
| MaxLevelCell | last active layer for cell | - |
| MinLevelEdgeTop | min of the first active layers for cells on edge | - |
| MaxLevelEdgeTop | min of the last active layer for cells on edge | - |
| MinLevelEdgeBot | max of the first active layer for cells on edge | - |
| MaxLevelEdgeBot | max of the last active layer for cells on edge | - |
| MinLevelVertexTop | min of the first active layer for cells on vertex | - |
| MaxLevelVertexTop | min of the last active layer for cells on vertex | - |
| MinLevelVertexBot | max of the first active layer for cells on vertex | - |
| MaxLevelVertexBot | max of the last active layer for cells on vertex | - |
| VertCoordMovementWeights | weights to specify how total column thickness changes are distributed across layers | - |
| RefLayerThickness | reference layer thickness used to distributed total column thickness changes | m |
| BottomDepth | positive down distance from the reference geoid to the bottom | m |

### Configuration options

The vertical coordinate movement can be specified by the `MovementWeightType` option in the configuration file.
```yaml
omega:
   VertCoord:
      MovementWeightType: [Fixed,Uniform]
```
The option `Uniform` specifies that perturbations to the total pseudo-thickness of the water column is distributed evenly to all vertical layers.
This is similar to the standard "z-star" coordinate in Boussinesq models
The option `Fixed` means that total pseudo-thickness perturbations are confined to the top layer, while all others remain constant.
This is similar to the traditional "z" coordinate in Boussinesq models
