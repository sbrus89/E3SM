//===-- ocn/BarotropicState.cpp - ocean state methods -------------*- C++ -*-===//
//
// The BarotropicState class initializes the prognostic variables in OMEGA.
// It contains a method to update the time levels for each variable.
// It is meant to provide a container for passing (non-tracer) prognostic
// variables throughout the OMEGA tendency computation routines.
//
//===----------------------------------------------------------------------===//

#include "BarotropicState.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Error.h"
#include "Field.h"
#include "Halo.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "TimeStepper.h"

namespace OMEGA {

//------------------------------------------------------------------------------
// Construct a new local state

BarotropicState::BarotropicState(
    const std::string &Name_, //< [in] Name for new state
    HorzMesh *Mesh,           //< [in] HorzMesh for state
    Halo *MeshHalo_,          //< [in] Halo for Mesh
    const int NTimeLevels_    //< [in] number of time levels
) {

   // Retrieve mesh cell/edge/vertex totals from Decomp
   NCellsOwned = Mesh->NCellsOwned;
   NCellsAll   = Mesh->NCellsAll;
   NCellsSize  = Mesh->NCellsSize;

   NEdgesOwned = Mesh->NEdgesOwned;
   NEdgesAll   = Mesh->NEdgesAll;
   NEdgesSize  = Mesh->NEdgesSize;

   NTimeLevels = NTimeLevels_;

   MeshHalo = MeshHalo_;

   Name = Name_;

   CurTimeIndex = 0;

   // Allocate state device arrays
   NormalBarotropicVelocity.resize(NTimeLevels);
   BarotropicPressureAnomaly.resize(NTimeLevels);

   // Create device arrays and copy host data
   for (int I = 0; I < NTimeLevels; I++) {
      NormalBarotropicVelocity[I] = Array1DReal(
          "NormalBarotropicVelocity" + std::to_string(I), NEdgesSize);
      BarotropicPressureAnomaly[I] = Array1DReal(
          "BarotropicPressureAnomaly" + std::to_string(I), NCellsSize);

      deepCopy(NormalBarotropicVelocity[I], 0.);
      deepCopy(BarotropicPressureAnomaly[I], 0.);
   }

} // end state constructor


//------------------------------------------------------------------------------
// Destroys a local mesh and deallocates all arrays
BarotropicState::~BarotropicState() {

   // Kokkos arrays removed when no longer in scope

} // end destructor

//------------------------------------------------------------------------------
// Get normal barotropic velocity device array
Array1DReal BarotropicState::getNormalBarotropicVelocity(const I4 TimeLevel) const {
   const I4 TimeIndex = getTimeIndex(TimeLevel);
   return NormalBarotropicVelocity[TimeIndex];
}

//------------------------------------------------------------------------------
// Get barotropic pressure anomaly device array
Array1DReal BarotropicState::getBarotropicPressureAnomaly(const I4 TimeLevel) const {
   const I4 TimeIndex = getTimeIndex(TimeLevel);
   return BarotropicPressureAnomaly[TimeIndex];
}

//------------------------------------------------------------------------------
// Perform state halo exchange
// TimeLevel == [1:new, 0:current, -1:previous, -2:two times ago, ...]
void BarotropicState::exchangeHalo(const I4 TimeLevel) {

   const I4 TimeIndex = getTimeIndex(TimeLevel);

   MeshHalo->exchangeFullArrayHalo(NormalBarotropicVelocity[TimeIndex], OnEdge);
   MeshHalo->exchangeFullArrayHalo(BarotropicPressureAnomaly[TimeIndex],
                                   OnCell);

} // end exchangeHalo

//------------------------------------------------------------------------------
// Perform time level update
void BarotropicState::updateTimeLevels() {

   if (NTimeLevels == 1)
      ABORT_ERROR("BarotropicState: can't update time levels for NTimeLevels == 1");

   // Exchange halo
   exchangeHalo(1);

   // Update current time index for layer thickness and normal velocity
   CurTimeIndex = (CurTimeIndex + 1) % NTimeLevels;

} // end updateTimeLevels

//------------------------------------------------------------------------------
// Get time index from time level
// TimeLevel == [1:new, 0:current, -1:previous, -2:two times ago, ...]
I4 BarotropicState::getTimeIndex(const I4 TimeLevel) const {

   OMEGA_REQUIRE(NTimeLevels <= 1 ||
                     !(TimeLevel > 1 || (TimeLevel + NTimeLevels) <= 1),
                 "BarotropicState: Time level {} is out of range for NTimeLevels {}",
                 TimeLevel, NTimeLevels);

   return (TimeLevel + CurTimeIndex + NTimeLevels) % NTimeLevels;
} // end get time index

} // end namespace OMEGA

//===----------------------------------------------------------------------===//
