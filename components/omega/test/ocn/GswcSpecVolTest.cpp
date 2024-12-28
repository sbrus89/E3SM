//===-- Test driver for OMEGA GSW-C library -----------------------------*- C++
//-*-===/
//
/// \file
/// \brief Test driver for OMEGA GSW-C external library
///
/// This driver tests that the GSW-C library can be called
/// and returns expected value (as published in Roquet et al 2015)
//
//===-----------------------------------------------------------------------===/

#include "Tracers.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Dimension.h"
#include "Field.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "TimeStepper.h"
#include "mpi.h"

#include <iostream>
#include </home/ac.abarthel/E3SMv3/code/add-eos/components/omega/external/GSW-C/gswteos-10.h>

using namespace OMEGA;

// misc. variables for testing
const Real RefReal = 3.0;

//------------------------------------------------------------------------------
// The initialization routine for Teos10 testing. It calls various
// init routines, including the creation of the default decomposition.

I4 initTeosTest() {

   I4 Err = 0;

   // Initialize the Machine Environment class - this also creates
   // the default MachEnv. Then retrieve the default environment and
   // some needed data members.
   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   initLogging(DefEnv);

   // Open config file
   Config("Omega");
   Err = Config::readAll("omega.yml");
   if (Err != 0) {
      LOG_ERROR("Tracers: Error reading config file");
      return Err;
   }

   // Initialize the default time stepper
   Err = TimeStepper::init1();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default time stepper");
      return Err;
   }

   // Initialize the IO system
   Err = IO::init(DefComm);
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing parallel IO");
      return Err;
   }

   // Create the default decomposition (initializes the decomposition)
   Err = Decomp::init();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default decomposition");
      return Err;
   }

   // Initialize the default halo
   Err = Halo::init();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default halo");
      return Err;
   }

   // Initialize the default mesh
   Err = HorzMesh::init();
   if (Err != 0) {
      LOG_ERROR("Tracers: error initializing default mesh");
      return Err;
   }

   return 0;
}

//------------------------------------------------------------------------------
// The test driver for Teos10 library testing -> this tests calls the library 
// and reads in the TEOS10 constants.
//
int main(int argc, char *argv[]) {

   int RetVal = 0;
   I4 Err;
   I4 count = 0;

   // Initialize the global MPI environment
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   {

      // Call initialization routine
      Err = initTeosTest();
      if (Err != 0)
         LOG_ERROR("Tracers: Error initializing");

      // Get MPI vars if needed
      MachEnv *DefEnv = MachEnv::getDefault();
      MPI_Comm Comm   = DefEnv->getComm();
      I4 MyTask       = DefEnv->getMyTask();
      I4 NumTasks     = DefEnv->getNumTasks();
      bool IsMaster   = DefEnv->isMasterTask();

      HorzMesh *DefHorzMesh       = HorzMesh::getDefault();
      Decomp *DefDecomp           = Decomp::getDefault();
      Halo *DefHalo               = Halo::getDefault();
      TimeStepper *DefTimeStepper = TimeStepper::getDefault();

      
      std::cout << "testing the teos test" << std::endl;
      // GSW_TEOS10_CONSTANTS;
      double print_specvol_test() {
        double sa = 30.;
        double ct = 10.;
	double p = 1000.;
        double delta = gsw_specvol(sa, ct, p);

        return delta;
      }

      // initialize Tracers infrastructure
      Err = Tracers::init();
      if (Err != 0) {
         RetVal += 1;
         LOG_ERROR("Tracers: initialzation FAIL");
      }

      I4 NTracers    = Tracers::getNumTracers();
      I4 NCellsOwned = DefHorzMesh->NCellsOwned;
      I4 NCellsSize  = DefHorzMesh->NCellsSize;
      I4 NVertLevels = DefHorzMesh->NVertLevels;
      I4 NTimeLevels = DefTimeStepper->getNTimeLevels();


      Tracers::clear();
      TimeStepper::clear();
      HorzMesh::clear();
      Decomp::clear();
      MachEnv::removeAll();
      FieldGroup::clear();
      Field::clear();
      Dimension::clear();

      if (RetVal == 0)
         LOG_INFO("Teos lib call: Successful completion");
   }
   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
