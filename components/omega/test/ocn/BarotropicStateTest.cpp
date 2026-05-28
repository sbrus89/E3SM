//===-- Test driver for OMEGA BarotropicState ------------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA BarotropicState class
///
/// This driver tests that the OMEGA BarotropicState class is correctly
/// constructed, zero-initialized, and that the time level update works as
/// expected.
//
//===-----------------------------------------------------------------------===/

#include "BarotropicState.h"
#include "Config.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Dimension.h"
#include "Error.h"
#include "Field.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "IOStream.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "Pacer.h"
#include "TimeStepper.h"
#include "VertCoord.h"
#include "mpi.h"

using namespace OMEGA;

//------------------------------------------------------------------------------
// Initialization routine for BarotropicState testing.

void initBarotropicStateTest() {

   int Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   initLogging(DefEnv);

   Config("Omega");
   Config::readAll("omega.yml");

   TimeStepper::init1();
   TimeStepper *DefStepper = TimeStepper::getDefault();
   Clock *ModelClock       = DefStepper->getClock();

   IO::init(DefComm);
   IOStream::init(ModelClock);
   Field::init(ModelClock);

   Decomp::init();

   Err = Halo::init();
   if (Err != 0)
      ABORT_ERROR("BarotropicState: error initializing default halo");

   HorzMesh::init();
   VertCoord::init();
}

//------------------------------------------------------------------------------
// Count differences between barotropic state device arrays at two time levels.

int checkDevice(BarotropicState *RefState, BarotropicState *TstState,
                int RefTimeLevel, int TstTimeLevel) {

   int count1;
   Array1DReal VelRef = RefState->getNormalBarotropicVelocity(RefTimeLevel);
   Array1DReal VelTst = TstState->getNormalBarotropicVelocity(TstTimeLevel);
   parallelReduce(
       "reduceVel", {RefState->NEdgesAll},
       KOKKOS_LAMBDA(int Edge, int &Accum) {
          if (VelRef(Edge) != VelTst(Edge))
             Accum++;
       },
       count1);

   int count2;
   Array1DReal PressRef =
       RefState->getBarotropicPressureAnomaly(RefTimeLevel);
   Array1DReal PressTst =
       TstState->getBarotropicPressureAnomaly(TstTimeLevel);
   parallelReduce(
       "reducePress", {RefState->NCellsAll},
       KOKKOS_LAMBDA(int Cell, int &Accum) {
          if (PressRef(Cell) != PressTst(Cell))
             Accum++;
       },
       count2);

   return count1 + count2;
}

//------------------------------------------------------------------------------
// Test driver for BarotropicState: tests construction, zero initialization,
// and time level updates.

int main(int argc, char *argv[]) {

   int RetVal = 0;

   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   Pacer::initialize(MPI_COMM_WORLD);
   Pacer::setPrefix("Omega:");
   {
      initBarotropicStateTest();

      HorzMesh *DefHorzMesh = HorzMesh::getDefault();
      Halo *DefHalo         = Halo::getDefault();
      int CurTime           = 0;
      int NewTime           = 1;

      // Test construction and zero initialization
      {
         BarotropicState State("TestState", DefHorzMesh, DefHalo, 2);
         int NCellsAll = State.NCellsAll;
         int NEdgesAll = State.NEdgesAll;

         if (State.NTimeLevels == 2 and NCellsAll > 0 and NEdgesAll > 0) {
            LOG_INFO("BarotropicState: Construction PASS");
         } else {
            RetVal += 1;
            LOG_INFO("BarotropicState: Construction FAIL");
         }

         Array1DReal VelDef   = State.getNormalBarotropicVelocity(CurTime);
         Array1DReal PressDef = State.getBarotropicPressureAnomaly(CurTime);

         int Count1 = 0;
         parallelReduce(
             "checkVelZero", {NEdgesAll},
             KOKKOS_LAMBDA(int Edge, int &Accum) {
                if (VelDef(Edge) != 0.0)
                   Accum++;
             },
             Count1);

         int Count2 = 0;
         parallelReduce(
             "checkPressZero", {NCellsAll},
             KOKKOS_LAMBDA(int Cell, int &Accum) {
                if (PressDef(Cell) != 0.0)
                   Accum++;
             },
             Count2);

         if (Count1 == 0 and Count2 == 0) {
            LOG_INFO("BarotropicState: Zero initialization PASS");
         } else {
            RetVal += 1;
            LOG_INFO("BarotropicState: Zero initialization FAIL");
         }
      }

      // Test time swapping with 2 and higher numbers of time levels
      for (int NTimeLevels = 2; NTimeLevels < 5; NTimeLevels++) {

         BarotropicState RefState("Reference", DefHorzMesh, DefHalo,
                                  NTimeLevels);
         BarotropicState TstState("Test", DefHorzMesh, DefHalo, NTimeLevels);

         if (RefState.NTimeLevels == NTimeLevels and
             TstState.NTimeLevels == NTimeLevels) {
            LOG_INFO("BarotropicState: State creation (NTimeLevels={}) PASS",
                     NTimeLevels);
         } else {
            RetVal += 1;
            LOG_INFO("BarotropicState: State creation (NTimeLevels={}) FAIL",
                     NTimeLevels);
         }

         // Fill reference and test states at CurTime with value 1.0
         deepCopy(RefState.getNormalBarotropicVelocity(CurTime), 1.0);
         deepCopy(TstState.getNormalBarotropicVelocity(CurTime), 1.0);
         deepCopy(RefState.getBarotropicPressureAnomaly(CurTime), 1.0);
         deepCopy(TstState.getBarotropicPressureAnomaly(CurTime), 1.0);

         // Fill reference and test states at NewTime with value 2.0
         deepCopy(RefState.getNormalBarotropicVelocity(NewTime), 2.0);
         deepCopy(TstState.getNormalBarotropicVelocity(NewTime), 2.0);
         deepCopy(RefState.getBarotropicPressureAnomaly(NewTime), 2.0);
         deepCopy(TstState.getBarotropicPressureAnomaly(NewTime), 2.0);

         // Check initial values match between ref and test states
         for (int N = 0; N <= 1; ++N) {
            int Count = checkDevice(&RefState, &TstState, N, N);
            if (Count == 0) {
               LOG_INFO("BarotropicState: State compare "
                        "(TimeLevel {}, NTimeLevels {}) PASS",
                        N, NTimeLevels);
            } else {
               RetVal += 1;
               LOG_INFO("BarotropicState: State compare "
                        "(TimeLevel {}, NTimeLevels {}) FAIL",
                        N, NTimeLevels);
            }
         }

         // Perform time level updates and verify time index rotation.
         // After each update, CurTimeIndex advances by 1 (mod NTimeLevels),
         // so the original CurTime and NewTime data appear at shifted indices.
         for (int N = 1; N < NTimeLevels; ++N) {
            TstState.updateTimeLevels();

            // The time index represents the n+Ith level: new=1, current=0,
            // previous=-1, etc. After N updates, indices shift by -N and wrap
            // at the lower bound -(NTimeLevels-2).
            int NMin          = -(NTimeLevels - 2);
            int CurTimeUpdate = CurTime - N;
            int NewTimeUpdate = NewTime - N;
            if (CurTimeUpdate < NMin)
               CurTimeUpdate += NTimeLevels;
            if (NewTimeUpdate < NMin)
               NewTimeUpdate += NTimeLevels;

            int Count = checkDevice(&RefState, &TstState, CurTime, CurTimeUpdate);
            if (Count == 0) {
               LOG_INFO("BarotropicState: NTimeLevels={} After update {} "
                        "Current level: PASS",
                        NTimeLevels, N);
            } else {
               RetVal += 1;
               LOG_INFO("BarotropicState: NTimeLevels={} After update {} "
                        "Current level: FAIL",
                        NTimeLevels, N);
            }

            Count = checkDevice(&RefState, &TstState, NewTime, NewTimeUpdate);
            if (Count == 0) {
               LOG_INFO("BarotropicState: NTimeLevels={} After update {} "
                        "New time level: PASS",
                        NTimeLevels, N);
            } else {
               RetVal += 1;
               LOG_INFO("BarotropicState: NTimeLevels={} After update {} "
                        "New time level: FAIL",
                        NTimeLevels, N);
            }
         }
         // RefState and TstState destructors called at end of loop body
      }

      // Finalize Omega objects
      TimeStepper::clear();
      HorzMesh::clear();
      VertCoord::clear();
      Halo::clear();
      Decomp::clear();
      MachEnv::removeAll();
      FieldGroup::clear();
      Field::clear();
      Dimension::clear();

      if (RetVal == 0)
         LOG_INFO("BarotropicState: Successful completion");
   }
   Pacer::finalize();
   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
