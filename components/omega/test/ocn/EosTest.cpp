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
#include "Config.h"
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
#include "Eos.h"

// added for debug
#include "Field.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "AuxiliaryState.h"

#include <gswteos-10.h>

using namespace OMEGA;


// struct TestSetup {
//    Real Radius = 6371220;
//
//    KOKKOS_FUNCTION Real layerThickness(Real Lon, Real Lat) const {
//       return (2 + std::cos(Lon) * std::pow(std::cos(Lat), 4));
//    }
//
//    KOKKOS_FUNCTION Real tracer(Real Lon, Real Lat) const {
//       return (2 - std::cos(Lon) * std::pow(std::cos(Lat), 4));
//    }
// };

constexpr Geometry Geom   = Geometry::Spherical;
constexpr int NVertLevels = 60;
// published values (TEOS-10) to test against
const Real DeltaRefReal = 0.0009776149797;
const Real VolRefReal = 0.0009732819628;
double Sa = 30.;
double Ct = 10.;
double P = 1000.;

//------------------------------------------------------------------------------
// The initialization routine for Eos testing. It calls various
// init routines, including the creation of the default decomposition.
// Not used now
I4 initEosTest(const std::string &mesh) {

   I4 Err = 0;

   // Initialize the Machine Environment class - this also creates
   // the default MachEnv. Then retrieve the default environment and
   // some needed data members.
   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   initLogging(DefEnv);
   LOG_INFO("… 4 in initEosTest");

   // Open config file
   Config("Omega");
   Err = Config::readAll("omega.yml");
   if (Err != 0) {
      LOG_ERROR("Eos: Error reading config file");
      return Err;
   }

   int TimeStepperErr = TimeStepper::init1();
   if (TimeStepperErr != 0) {
      Err++;
      LOG_ERROR("TendenciesTest: error initializing default time stepper");
   }

   int IOErr = IO::init(DefComm);
   if (IOErr != 0) {
      Err++;
      LOG_ERROR("TendenciesTest: error initializing parallel IO");
   }

   int DecompErr = Decomp::init(mesh);
   if (DecompErr != 0) {
      Err++;
      LOG_ERROR("TendenciesTest: error initializing default decomposition");
   }

   int HaloErr = Halo::init();
   if (HaloErr != 0) {
      Err++;
      LOG_ERROR("TendenciesTest: error initializing default halo");
   }

   int MeshErr = HorzMesh::init();
   if (MeshErr != 0) {
      Err++;
      LOG_ERROR("TendenciesTest: error initializing default mesh");
   }

   int TracerErr = Tracers::init();
   if (TracerErr != 0) {
      Err++;
      LOG_ERROR("TendenciesTest: error initializing tracer infrastructure");
   }

   const auto &Mesh = HorzMesh::getDefault();
   std::shared_ptr<Dimension> VertDim =
       Dimension::create("NVertLevels", NVertLevels);


   return Err;
}


int testEos() {
   int Err = 0;
   LOG_INFO("… in EosTest");
   // test initialization
   int EosErr = Eos::init();
   if (EosErr != 0) {
      Err++;
      LOG_ERROR("EosTest: error initializing default Eos");
   }

   // test retrievel of default
   Eos *DefEos = Eos::getDefault();

   if (DefEos) {
      LOG_INFO("EosTest: Default Eos retrieval PASS");
   } else {
      Err++;
      LOG_INFO("EosTest: Default Eos retrieval FAIL");
      return -1;
   }

   const auto *Mesh = HorzMesh::getDefault();
   // test creation of another Eos
   Config *Options = Config::getOmegaConfig();
   Eos::create("TestEos", Mesh, 60); // MODIFY THIS

   // test retrievel of another Eos
   if (Eos::get("TestEos")) {
      LOG_INFO("EosTest: Non-default Eos retrieval PASS");
   } else {
      Err++;
      LOG_INFO("EosTest: Non-default Eos retrieval FAIL");
   }

// test erase
   Eos::erase("TestEos");

   if (Eos::get("TestEos")) {
      Err++;
      LOG_INFO("EosTest: Non-default Eos erase FAIL");
   } else {
      LOG_INFO("EosTest: Non-default Eos erase PASS");
   }

   Eos::clear();

   return Err;
}

void finalizeEosTest() {
   Tracers::clear();
   AuxiliaryState::clear();
   HorzMesh::clear();
   Halo::clear();
   TimeStepper::clear();
   Decomp::clear();
   MachEnv::removeAll();
   LOG_INFO("EosTest: end of finalize()");
}

int eosTest(const std::string &MeshFile = "OmegaMesh.nc"){
   int Err = initEosTest(MeshFile);
   if (Err != 0) {
      LOG_CRITICAL("EosTest: Error initializing");
   }
   const auto &Mesh = HorzMesh::getDefault();
   Err += testEos();

   if (Err == 0) {
      LOG_INFO("EosTest: Successful completion");
   }
   finalizeEosTest();

   return Err;
}


int gswcSpecVolCheckValue() {
   int Err = 0;
   const Real RTol = 1e-10;

   double SpecVol = gsw_specvol(Sa, Ct, P);
   LOG_INFO("gswcSpecVolCheckValue: produced SpecVol from GSW-C module");
   LOG_INFO("Value of SpecVol: {}", SpecVol);
   bool Check = isApprox(SpecVol, VolRefReal, RTol);
   if (!Check) {
      Err++;
      LOG_ERROR("gswcSpecVolCheckValue: SpecVol isApprox FAIL, expected {}, got {}",
                VolRefReal, SpecVol);
   }
   if (Err == 0) {
      LOG_INFO("gswcSpecVolCheckValue: PASS");
   }
   return Err;
}

// intermediate test: not used for now
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

int poly75tDeltaCheckValue() {
   int Err = 0;
   const Real RTol = 1e-10;

   TEOS10Poly75t specvolpoly75t;
   Real Delta = specvolpoly75t.calcdelta(Sa, Ct, P);
   LOG_INFO("Teos10 poly75tDeltaCheckValue: produced delta from poly75t");
   LOG_INFO("Value of Delta: {}", Delta);
   bool Check = isApprox(Delta, DeltaRefReal, RTol);
   if (!Check) {
      Err++;
      LOG_ERROR("Teos10 poly75tDeltaCheckValue: Delta isApprox FAIL, expected {}, got {}",
                DeltaRefReal, Delta);
   }
   if (Err == 0) {
      LOG_INFO("Teos10 poly75tDeltaCheckValue: PASS");
   }
   return Err;
}

int poly75tSpecVolCheckValue() {
   int Err = 0;
   const Real RTol = 1e-10;

   TEOS10Poly75t specvolpoly75t;
   Real SpecVol = specvolpoly75t(Sa, Ct, P);
   LOG_INFO("Teos10 poly75tSpecVolCheckValue: produced SpecVol from poly75t");
   LOG_INFO("Value of SpecVol: {}", SpecVol);
   bool Check = isApprox(SpecVol, VolRefReal, RTol);
   if (!Check) {
      Err++;
      LOG_ERROR("Teos10 poly75tSpecVolCheckValue: SpecVol isApprox FAIL, expected {}, got {}",
                VolRefReal, SpecVol);
   }
   if (Err == 0) {
      LOG_INFO("Teos10 poly75tSpecVolCheckValue: PASS");
   }
   return Err;
}

int linearSpecVolCheckValue() {
   int Err = 0;
   const Real RTol = 1e-10;
   const Real RTol2 = 1e-2; // >>machine prection
   Real Sa = 30.;
   Real Ct = 10.;
   Real P = 1000.;

   LinearEOS specvollinear;
   Real SpecVol = specvollinear(Sa, Ct, P);
   LOG_INFO("linearSpecVolCheckValue: produced SpecVol from linear EOS");
   LOG_INFO("Value of SpecVol: {}", SpecVol);
   bool Check = isApprox(SpecVol, VolRefReal, RTol); // expect False
   bool CheckClose = isApprox(SpecVol, VolRefReal, RTol2); // expect True
   if (Check) {
      Err++;
      LOG_ERROR("linearSpecVolCheckValue: SpecVol Linear is undistinguishable from TEOS10 Ref Value");
   }
   else if (!Check) {
      LOG_INFO("linearSpecVolCheckValue: SpecVol TEOS10 {}, got {} with Linear",
                VolRefReal, SpecVol);
   }
   if (CheckClose) {
      LOG_INFO("linearSpecVolCheckValue: SpecVol TEOS10 and Linear are within {}",
		RTol2);
   }
   else if (!CheckClose) {
      Err++;
      LOG_ERROR("linearSpecVolCheckValue: SpecVol TEOS10 and Linear are NOT close. Check input values");
   }
   if (Err == 0) {
      LOG_INFO("linearSpecVolCheckValue: PASS");
   }
   return Err;
}

int linearDensityLinearityTest() {
   int Err = 0;
   const Real RTol = 1e-10;
   Real Sa1 = 30.;
   Real Ct1 = 15.;
   Real Sa2 = 33.;
   Real Ct2 = 10.;

   LinearEOS specvollinear;
   Real DRhoS = 1./specvollinear(Sa2, Ct1, P) - 1./specvollinear(Sa1, Ct1, P);
   Real DRhoT = 1./specvollinear(Sa1, Ct2, P) - 1./specvollinear(Sa1, Ct1, P);
   Real DRhoTS = 1./specvollinear(Sa2, Ct2, P) - 1./specvollinear(Sa1, Ct1, P);
   LOG_INFO("linearDensityLinearityTest: produced SpecVol from linear EOS");
   LOG_INFO("Value of drhodS: {}", DRhoS);
   LOG_INFO("Value of drhodT: {}", DRhoT);
   bool Check = isApprox(DRhoTS, DRhoS + DRhoT, RTol);
   if (!Check) {
      Err++;
      LOG_ERROR("linearDensityLinearityTest: Sum(DRho) {}, DRhoTS {}",
		 DRhoS + DRhoT, DRhoTS);
   }
   LOG_INFO("linearDensityLinearityTest: Sum(DRho) is undistinguishable from DRhoTS");
   if (Err == 0) {
      LOG_INFO("linearDensityLinearityTest: PASS");
   }
   return Err;
}

//------------------------------------------------------------------------------
// The test driver for Eos testing
// --> one test calls the external GSW-C library
// and compares the specific volume to the published value
// --> next tests call the local POly75t TEOS-10 calc
// --> next tests call the linear Eos
int main(int argc, char *argv[]) {

   int RetVal = 0;

   MPI_Init(&argc, &argv);
   Kokkos::initialize(argc, argv);
   {
//    RetVal += gswcSpecVolCheckValue();
//    RetVal += poly75tDeltaCheckValue();
//    RetVal += poly75tSpecVolCheckValue();
//    RetVal += linearSpecVolCheckValue();
//    RetVal += linearDensityLinearityTest();
   RetVal += eosTest();
   }
   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
