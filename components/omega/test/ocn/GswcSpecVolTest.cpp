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
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OceanTestCommon.h"
#include "OmegaKokkos.h"
#include "TimeStepper.h"
#include "mpi.h"
#include "EosConstants.h"

#include <iostream>
#include <gswteos-10.h>

using namespace OMEGA;

// misc. variables for testing
const Real DeltaRefReal = 0.0009776149797;
const Real VolRefReal = 0.0009732819628;

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
      LOG_ERROR("Teos: Error reading config file");
      return Err;
   }

   // Initialize the IO system
   Err = IO::init(DefComm);
   if (Err != 0) {
      LOG_ERROR("Teos: error initializing parallel IO");
      return Err;
   }

   return 0;
}


int test_specvol_value() {
   int Err = 0;
   const Real RTol = 1e-10;
   double Sa = 30.;
   double Ct = 10.;
   double P = 1000.;
   
   double SpecVol = gsw_specvol(Sa, Ct, P);
   LOG_INFO("GswcTeosTest: produced SpecVol from GSW-C module");
   LOG_INFO("Value of SpecVol: {}", SpecVol);
   bool Check = isApprox(SpecVol, VolRefReal, RTol);
   if (!Check) {
      Err++;
      LOG_ERROR("GswcTeosTest: SpecVol isApprox FAIL, expected {}, got {}",
                VolRefReal, SpecVol);
   }
   if (Err == 0) {
      LOG_INFO("GswcTeosTest: check PASS");
   }
   return Err;
}

int test_fetch_coeff() {
   int Err = 0;
   double ExpVal = 0.0010769995862;
   const Real RTol = 1e-10;
   GSW_SPECVOL_COEFFICIENTS;
   LOG_INFO("EosTest: called GSW_SPECVOL_COEFFICIENTS");
   LOG_INFO("Value of V000: {}", V000);
   if (!isApprox(V000, ExpVal, RTol)) {
      Err++;
      LOG_ERROR("EosTest: Coeff V000 isApprox FAIL, expected {}, got {}",
                ExpVal, V000);
   }
   if (Err == 0) {
      LOG_INFO("GswcTeosTest: check PASS");
   }
   return Err;
}

//------------------------------------------------------------------------------
// The test driver for Teos10 library testing -> this tests calls the library 
// and compares the specific volume to the published value
//
int main(int argc, char *argv[]) {

   int RetVal = 0;

   MPI_Init(&argc, &argv);
   Kokkos::initialize(argc, argv);

   RetVal += test_specvol_value();
   RetVal += test_fetch_coeff();

   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
