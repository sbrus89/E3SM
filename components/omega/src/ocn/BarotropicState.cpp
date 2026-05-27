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

// create the static class members
BarotropicState *BarotropicState::DefaultBarotropicState = nullptr;
std::map<std::string, std::unique_ptr<BarotropicState>> BarotropicState::AllBarotropicStates;

//------------------------------------------------------------------------------
// Initialize the state. Assumes that Decomp, HorzMesh, VertCoord, and
// TimeStepper have already been initialized.

int BarotropicState::init() {

   int Err = 0; // default successful return code

   // Retrieve the default decomposition and mesh
   Decomp *DefDecomp     = Decomp::getDefault();
   HorzMesh *DefHorzMesh = HorzMesh::getDefault();
   Halo *DefHalo         = Halo::getDefault();

   auto *DefTimeStepper = TimeStepper::getDefault();
   if (!DefTimeStepper) {
      LOG_ERROR("TimeStepper needs to be initialized before BarotropicState");
   }
   int NTimeLevels = DefTimeStepper->getNTimeLevels();
   LOG_INFO("BarotropicState: Initializing default state with {} time levels",
             NTimeLevels);

   if (NTimeLevels < 2) {
      LOG_ERROR("BarotropicState: the number of time level is lower than 2");
      return -2;
   }

   // Create the default state and set pointer to it
   BarotropicState::DefaultBarotropicState =
       create("Default", DefHorzMesh, DefHalo, NTimeLevels);

   // State values are filled by a later read of the initial state or
   // restart file

   return Err;
}

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

/// Create a new state by calling the constructor and put it in the
/// AllBarotropicStates map
BarotropicState *
BarotropicState::create(const std::string &Name, //< [in] Name for new state
                        HorzMesh *Mesh,          //< [in] HorzMesh for state
                        Halo *MeshHalo,          //< [in] Halo for Mesh
                        const int NTimeLevels    //< [in] number of time levels
) {

   // Check to see if a state of the same name already exists and
   // if so, exit with an error
   if (AllBarotropicStates.find(Name) != AllBarotropicStates.end()) {
      LOG_ERROR(
          "Attempted to create an BarotropicState with name {} but an BarotropicState of "
          "that name already exists",
          Name);
      return nullptr;
   }

   // create a new state on the heap and put it in a map of
   // unique_ptrs, which will manage its lifetime
   auto *NewBarotropicState =
       new BarotropicState(Name, Mesh, MeshHalo, NTimeLevels);
   AllBarotropicStates.emplace(Name, NewBarotropicState);

   return NewBarotropicState;
} // end state create

//------------------------------------------------------------------------------
// Destroys a local mesh and deallocates all arrays
BarotropicState::~BarotropicState() {

   // Kokkos arrays removed when no longer in scope

} // end destructor

//------------------------------------------------------------------------------
// Removes a state from list by name
void BarotropicState::erase(std::string InName // [in] name of state to remove
) {

   AllBarotropicStates.erase(InName); // remove the state from the list and in
                                 // the process, calls the destructor

} // end state erase
//------------------------------------------------------------------------------
// Removes all states to clean up before exit
void BarotropicState::clear() {

   AllBarotropicStates.clear();      // removes all states from the list and in
                                // the process, calls the destructors for each
   DefaultBarotropicState = nullptr; // prevent dangling pointer
} // end clear


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
// Get default state
BarotropicState *BarotropicState::getDefault() { return BarotropicState::DefaultBarotropicState; }

//------------------------------------------------------------------------------
// Get state by name
BarotropicState *BarotropicState::get(const std::string Name ///< [in] Name of state
) {

   // look for an instance of this name
   auto it = AllBarotropicStates.find(Name);

   // if found, return the state pointer
   if (it != AllBarotropicStates.end()) {
      return it->second.get();

      // otherwise print error and return null pointer
   } else {
      LOG_ERROR("BarotropicState::get: Attempt to retrieve non-existent state:");
      LOG_ERROR("{} has not been defined or has been removed", Name);
      return nullptr;
   }
} // end get state

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
