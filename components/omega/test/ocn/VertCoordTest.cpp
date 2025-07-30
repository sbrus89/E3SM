//===-- Test driver for OMEGA Vertical Coordinate ----------------*- C++ -*-===/
//
/// \file
/// \brief Test driver for OMEGA VertCoord class
///
///
//
//===-----------------------------------------------------------------------===/

#include "VertCoord.h"
#include "DataTypes.h"
#include "Decomp.h"
#include "Dimension.h"
#include "Error.h"
#include "Halo.h"
#include "HorzMesh.h"
#include "IO.h"
#include "Logging.h"
#include "MachEnv.h"
#include "OmegaKokkos.h"
#include "Pacer.h"
#include "mpi.h"

#include <algorithm>
#include <iostream>

using namespace OMEGA;

int initVertCoordTest() {

   int Err = 0;

   MachEnv::init(MPI_COMM_WORLD);
   MachEnv *DefEnv  = MachEnv::getDefault();
   MPI_Comm DefComm = DefEnv->getComm();

   // Initialize the Logging system
   initLogging(DefEnv);

   // Open config file
   Config("Omega");
   Config::readAll("omega.yml");

   // Initialize the IO system
   Err = IO::init(DefComm);
   if (Err != 0)
      LOG_ERROR("HorzMeshTest: error initializing parallel IO");

   // Create the default decomposition (initializes the decomposition)
   Decomp::init();

   // Initialize the default halo
   Err = Halo::init();
   if (Err != 0)
      LOG_ERROR("HorzMeshTest: error initializing default halo");

   // Initialize the default mesh
   HorzMesh::init();

   VertCoord::init();

   return Err;
} // end initVertCoordTest

//------------------------------------------------------------------------------
// The test driver for VertCoord test
//
int main(int argc, char *argv[]) {

   int RetVal = 0;

   // Initialize the global MPI environment
   MPI_Init(&argc, &argv);
   Kokkos::initialize();
   Pacer::initialize(MPI_COMM_WORLD);
   Pacer::setPrefix("Omega:");
   {
      int Err = initVertCoordTest();
      if (Err != 0)
         LOG_CRITICAL("VertCoordTest: Error initializing");

      auto *DefVertCoord = VertCoord::getDefault();
      auto *DefMesh      = HorzMesh::getDefault();

      I4 NCellsSize    = DefMesh->NCellsSize;
      I4 NCellsAll     = DefMesh->NCellsAll;
      I4 NEdgesAll     = DefMesh->NEdgesAll;
      I4 NVerticesAll  = DefMesh->NVerticesAll;
      I4 VertexDegree  = DefMesh->VertexDegree;
      I4 NVertLevels   = DefVertCoord->NVertLevels;
      I4 NVertLevelsP1 = DefVertCoord->NVertLevelsP1;

      // Tests for computePressure

      Array2DReal LayerThickness("LayerThickness", NCellsSize, NVertLevels);
      Array2DReal PressureInterface("PressureInterface", NCellsSize,
                                    NVertLevelsP1);
      Array2DReal PressureMid("PressureMid", NCellsSize, NVertLevels);
      Array1DReal SurfacePressure("SurfacePressure", NCellsSize);

      /// Initialize layer thickness and surface pressure so that resulting
      /// interface pressure is the number of layers above plus one
      Real Gravity = 9.80616_Real;
      Real Rho0    = 1035._Real;
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             SurfacePressure(ICell) = 1.0_Real;
             for (int K = 0; K < NVertLevels; K++) {
                LayerThickness(ICell, K) = 1.0_Real / (Gravity * Rho0);
             }
          });
      Kokkos::fence();

      /// Call function and get host copies of outputs
      DefVertCoord->computePressure(PressureInterface, PressureMid,
                                    LayerThickness, SurfacePressure);
      auto PressureInterfaceH = createHostMirrorCopy(PressureInterface);
      auto PressureMidH       = createHostMirrorCopy(PressureMid);

      /// Check results
      Err = 0;
      for (int ICell = 0; ICell < NCellsAll; ICell++) {
         for (int K = DefVertCoord->MinLevelCellH(ICell);
              K < DefVertCoord->MaxLevelCellH(ICell) + 1; K++) {
            // Interface pressure at level K should be K+1
            Real Expected = K + 1;
            Real Diff     = std::abs(PressureInterfaceH(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += 1;
            }
            // Mid pressures at level K should be K+1.5
            Expected = K + 1.5;
            Diff     = std::abs(PressureMidH(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += Err + 1;
            }
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO(
             "VertCoordTest: computePressure with uniform LayerThickness PASS");
      } else {
         LOG_INFO(
             "VertCoordTest: computePressure with uniform LayerThickness FAIL");
         RetVal += 1;
      }

      /// Initialize layer thickness and surface pressure so that the resulting
      /// interface pressure is (K+1)*K/2 + the cell number
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             SurfacePressure(ICell) = 1.0_Real * ICell;
             for (int K = 0; K < NVertLevels; K++) {
                LayerThickness(ICell, K) = (K + 1.0_Real) / (Gravity * Rho0);
             }
          });
      Kokkos::fence();

      /// Call functions and get host copy of output
      DefVertCoord->computePressure(PressureInterface, PressureMid,
                                    LayerThickness, SurfacePressure);
      auto PressureInterfaceH2 = createHostMirrorCopy(PressureInterface);

      /// Check results
      Err = 0;
      for (int ICell = 0; ICell < NCellsAll; ICell++) {
         for (int K = DefVertCoord->MinLevelCellH(ICell);
              K < DefVertCoord->MaxLevelCellH(ICell) + 1; K++) {
            /// Interface pressure should be (K+1)*K/2 + the cell number
            Real Expected = ((K + 1.0_Real) * K) / 2.0_Real + ICell;
            Real Diff     = std::abs(PressureInterfaceH2(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += 1;
            }
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO("VertCoordTest: computePressure with non-uniform "
                  "LayerThickness PASS");
      } else {
         LOG_INFO("VertCoordTest: computePressure with non-uniform "
                  "LayerThickness FAIL");
         RetVal += 1;
      }

      // Tests for computeZHeight

      Array2DReal ZInterface("ZInterface", NCellsSize, NVertLevelsP1);
      Array2DReal ZMid("ZMid", NCellsSize, NVertLevels);
      Array2DReal SpecVol("SpecVol", NCellsSize, NVertLevels);
      Array1DReal BottomDepth("BottomDepth", NCellsSize);
      Array1DReal MaxLevelCell("MaxLevelCell", NCellsSize);
      deepCopy(MaxLevelCell, DefVertCoord->MaxLevelCell);

      /// Initialize bottom depth, layer thickness and specific volume so that
      /// the resulting interface z value is the negative layer number
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             BottomDepth(ICell) = MaxLevelCell(ICell) + 1.0_Real;
             for (int K = 0; K < NVertLevels; K++) {
                LayerThickness(ICell, K) = (ICell + 1.0_Real) / Rho0;
                SpecVol(ICell, K)        = 1.0_Real / (ICell + 1.0_Real);
             }
          });
      Kokkos::fence();

      /// Call functions and get host copy of output
      DefVertCoord->computeZHeight(ZInterface, ZMid, LayerThickness, SpecVol,
                                   BottomDepth);
      auto ZInterfaceH = createHostMirrorCopy(ZInterface);
      auto ZMidH       = createHostMirrorCopy(ZMid);

      /// Check results
      Err = 0;
      for (int ICell = 0; ICell < NCellsAll; ICell++) {
         for (int K = DefVertCoord->MinLevelCellH(ICell);
              K < DefVertCoord->MaxLevelCellH(ICell) + 1; K++) {
            /// Z value at interface K should be -K
            Real Expected = -K;
            Real Diff     = std::abs(ZInterfaceH(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += 1;
            }
            /// Z value at mid point of layer K should be -(K + .5)
            Expected = -K - 0.5;
            Diff     = std::abs(ZMidH(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += 1;
            }
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO(
             "VertCoordTest: computeZHeight with uniform LayerThickness PASS");
      } else {
         LOG_INFO(
             "VertCoordTest: computeZHeight with uniform LayerThickness FAIL");
         RetVal += 1;
      }

      /// Initialize bottom depth, layer thickness and specific volume so that
      /// the resulting interface z value is -(K+1)*K/2
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             BottomDepth(ICell) = (MaxLevelCell(ICell) + 2) *
                                  (MaxLevelCell(ICell) + 1) / 2.0_Real;
             for (int K = 0; K < NVertLevels; K++) {
                LayerThickness(ICell, K) = (K + 1) / Rho0;
                SpecVol(ICell, K)        = 1.0_Real;
             }
          });
      Kokkos::fence();

      /// Call functions and get host copy of output
      DefVertCoord->computeZHeight(ZInterface, ZMid, LayerThickness, SpecVol,
                                   BottomDepth);
      auto ZInterfaceH2 = createHostMirrorCopy(ZInterface);

      /// Check results
      Err = 0;
      for (int ICell = 0; ICell < NCellsAll; ICell++) {
         for (int K = DefVertCoord->MinLevelCellH(ICell);
              K < DefVertCoord->MaxLevelCellH(ICell) + 1; K++) {
            /// Z value at interface should be -(K+1)*K/2
            Real Expected = -((K + 1.0_Real) * K) / 2.0_Real;
            Real Diff     = std::abs(ZInterfaceH2(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += 1;
            }
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO("VertCoordTest: computeZHeight with non-uniform "
                  "LayerThickness PASS");
      } else {
         LOG_INFO("VertCoordTest: computeZHeight with non-uniform "
                  "LayerThickness FAIL");
         RetVal += 1;
      }

      // Tests for computeGeopotential
      Array2DReal GeopotentialMid("GeopotentialMid", NCellsSize, NVertLevels);
      Array1DReal TidalPotential("TidalPotential", NCellsSize);
      Array1DReal SelfAttractionLoading("SelfAttractionLoading", NCellsSize);

      /// Initialize z mid, tidal potential and SAL so that the resulting
      /// geopotential is the cell number + level number
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             TidalPotential(ICell)        = ICell;
             SelfAttractionLoading(ICell) = -ICell;
             for (int K = 0; K < NVertLevels; K++) {
                ZMid(ICell, K) = (ICell + K) / Gravity;
             }
          });
      Kokkos::fence();

      /// Call functions and get host copy of output
      DefVertCoord->computeGeopotential(GeopotentialMid, ZMid, TidalPotential,
                                        SelfAttractionLoading);
      auto GeopotentialMidH = createHostMirrorCopy(GeopotentialMid);

      /// Check results
      Err = 0;
      for (int ICell = 0; ICell < NCellsAll; ICell++) {
         for (int K = DefVertCoord->MinLevelCellH(ICell);
              K < DefVertCoord->MaxLevelCellH(ICell) + 1; K++) {
            /// Geopotential should be cell number + layer number
            Real Expected = ICell + K;
            Real Diff     = std::abs(GeopotentialMidH(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += 1;
            }
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO("VertCoordTest: computeGeopotential PASS");
      } else {
         LOG_INFO("VertCoordTest: computeGeopotential FAIL");
         RetVal += 1;
      }

      // Tests for computePStarThickness
      Array2DReal LayerThicknessPStar("LayerThicknessPStar", NCellsSize,
                                      NVertLevels);
      Array2DReal VertCoordMovementWeights("VertCoordMovementWeights",
                                           NCellsSize, NVertLevels);
      Array2DReal RefLayerThickness("RefLayerThickness", NCellsSize,
                                    NVertLevels);

      /// Initialize surface pressure, vertical coord weights, ref layer
      /// thickness, and layer thickness so that the resulting p star thickness
      /// is 2 (perturbation is evenly distributed amoung layers)
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             SurfacePressure(ICell) = 0.0;
             for (int K = 0; K < NVertLevels; K++) {
                VertCoordMovementWeights(ICell, K) = 1.0;
                RefLayerThickness(ICell, K)        = 1.0;
                LayerThickness(ICell, K)           = 2.0;
             }
          });
      Kokkos::fence();

      /// Call functions and get host copy of output
      DefVertCoord->computePressure(PressureInterface, PressureMid,
                                    LayerThickness, SurfacePressure);
      DefVertCoord->computePStarThickness(LayerThicknessPStar,
                                          VertCoordMovementWeights,
                                          RefLayerThickness, PressureInterface);
      auto LayerThicknessPStarH = createHostMirrorCopy(LayerThicknessPStar);

      /// Check results
      Err = 0;
      for (int ICell = 0; ICell < NCellsAll; ICell++) {
         for (int K = DefVertCoord->MinLevelCellH(ICell);
              K < DefVertCoord->MaxLevelCellH(ICell) + 1; K++) {
            /// p star thickness should be 2
            Real Expected = 2.0;
            Real Diff     = std::abs(LayerThicknessPStarH(ICell, K) - Expected);
            if (Diff > 1e-10) {
               Err += 1;
            }
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO("VertCoordTest: computePStarThickness with uniform "
                  "distribution PASS");
      } else {
         LOG_INFO("VertCoordTest: computePStarThickness with uniform "
                  "distribution FAIL");
         RetVal += 1;
      }

      /// Intialize surface pressure, vertical coord weights, ref layer
      /// thickness, and layer thickness so that the resulting p star thickness
      /// is the max number of levels + 2 in the top layer and 1 elsewhere
      /// (perturbation is distributed to top level only)
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             SurfacePressure(ICell) = 0.0;
             for (int K = 0; K < NVertLevels; K++) {
                VertCoordMovementWeights(ICell, K) = 0.0;
                RefLayerThickness(ICell, K)        = 1.0;
                LayerThickness(ICell, K)           = 2.0;
             }
             VertCoordMovementWeights(ICell, 0) = 1.0;
          });
      Kokkos::fence();

      /// Call functions and get host copy of output
      DefVertCoord->computePressure(PressureInterface, PressureMid,
                                    LayerThickness, SurfacePressure);
      DefVertCoord->computePStarThickness(LayerThicknessPStar,
                                          VertCoordMovementWeights,
                                          RefLayerThickness, PressureInterface);
      auto LayerThicknessPStarH2 = createHostMirrorCopy(LayerThicknessPStar);
      Err                        = 0;

      /// Check results
      for (int ICell = 0; ICell < NCellsAll; ICell++) {
         for (int K = DefVertCoord->MinLevelCellH(ICell);
              K < DefVertCoord->MaxLevelCellH(ICell) + 1; K++) {
            Real Expected;
            if (K == 0) {
               /// p star thickness is number of layers + 2 in top layer
               Expected = DefVertCoord->MaxLevelCellH(ICell) + 2;
            } else {
               /// p star thickness is 1 in all other layer
               Expected = 1.0;
            }
            Real Diff = std::abs(LayerThicknessPStarH2(ICell, K) - Expected);
            if (Diff > 1e-10) {
               LOG_INFO("LayerThicknessPStarH({},{}) = {}, {}", ICell, K,
                        LayerThicknessPStarH2(ICell, K), Expected);
               Err += 1;
            }
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO("VertCoordTest: computePStarThickness with top only "
                  "distribution PASS");
      } else {
         LOG_INFO("VertCoordTest: computePStarThickness with top only "
                  "distribution FAIL");
         RetVal += 1;
      }

      // Tests for minMaxLevelEdge

      /// Initialize min/max number of cell layers such that
      /// the cellsOnEdge information can be used to determine min/max level
      /// edge
      const auto &LocMinLevelCell = DefVertCoord->MinLevelCell;
      const auto &LocMaxLevelCell = DefVertCoord->MaxLevelCell;
      parallelFor(
          {NCellsAll}, KOKKOS_LAMBDA(int ICell) {
             LocMinLevelCell(ICell) = -2 * ICell;
             LocMaxLevelCell(ICell) = 2 * ICell;
          });
      Kokkos::fence();

      /// Call function, outputs are member variables of class
      DefVertCoord->minMaxLevelEdge();

      /// Check results
      Err = 0;
      for (int IEdge = 0; IEdge < NEdgesAll; IEdge++) {
         I4 Expected;
         I4 Count = 0;

         /// Skip edges on boundary
         if (DefMesh->CellsOnEdgeH(IEdge, 1) == NCellsAll) {
            continue;
         }

         /// MinLevelEdgeTop is the min of the min cell values on edge
         Expected  = std::min(-2 * DefMesh->CellsOnEdgeH(IEdge, 0),
                              -2 * DefMesh->CellsOnEdgeH(IEdge, 1));
         Real Diff = std::abs(DefVertCoord->MinLevelEdgeTopH(IEdge) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }
         /// MinLevelEdgeBot is the max of the min cell values on edge
         Expected = std::max(-2 * DefMesh->CellsOnEdgeH(IEdge, 0),
                             -2 * DefMesh->CellsOnEdgeH(IEdge, 1));
         Diff     = std::abs(DefVertCoord->MinLevelEdgeBotH(IEdge) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }
         /// MaxLevelEdgeTop is the min of the max cell values on edge
         Expected = std::min(2 * DefMesh->CellsOnEdgeH(IEdge, 0),
                             2 * DefMesh->CellsOnEdgeH(IEdge, 1));
         Diff     = std::abs(DefVertCoord->MaxLevelEdgeTopH(IEdge) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }
         /// MaxLevelEdgeBot is the max of the max cell values on edge
         Expected = std::max(2 * DefMesh->CellsOnEdgeH(IEdge, 0),
                             2 * DefMesh->CellsOnEdgeH(IEdge, 1));
         Diff     = std::abs(DefVertCoord->MaxLevelEdgeBotH(IEdge) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO("VertCoordTest: minMaxLevelEdge PASS");
      } else {
         LOG_INFO("VertCoordTest: minMaxLevelEdge FAIL");
         RetVal += 1;
      }

      // Tests for minMaxLevelVertex

      /// Use MinLevelCell, MaxLevelCell values initialized in previous test
      /// CellsOnVertex information can be used to determine min/max level
      /// vertex

      /// Call function, outputs are member variables of class
      DefVertCoord->minMaxLevelVertex();

      /// Check results
      Err = 0;
      for (int IVertex = 0; IVertex < NVerticesAll; IVertex++) {

         /// Skip vertices on boundary
         I4 Boundary = 0;
         for (int I = 0; I < VertexDegree; I++) {
            if (DefMesh->CellsOnVertexH(IVertex, I) == NCellsAll) {
               Boundary += 1;
            }
         }
         if (Boundary > 0) {
            continue;
         }

         /// MinLevelVertexTop is the min of the min cell values on vertex
         I4 Expected = 1e7;
         for (int I = 0; I < VertexDegree; I++) {
            Expected =
                std::min(Expected, -2 * DefMesh->CellsOnVertexH(IVertex, I));
         }
         Real Diff =
             std::abs(DefVertCoord->MinLevelVertexTopH(IVertex) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }

         /// MinLevelVertexBot is the max of the min cell values on vertex
         Expected = -1e7;
         for (int I = 0; I < VertexDegree; I++) {
            Expected =
                std::max(Expected, -2 * DefMesh->CellsOnVertexH(IVertex, I));
         }
         Diff = std::abs(DefVertCoord->MinLevelVertexBotH(IVertex) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }

         /// MaxLevelVertexTop is the min of the max cell values on vertex
         Expected = 1e7;
         for (int I = 0; I < VertexDegree; I++) {
            Expected =
                std::min(Expected, 2 * DefMesh->CellsOnVertexH(IVertex, I));
         }
         Diff = std::abs(DefVertCoord->MaxLevelVertexTopH(IVertex) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }

         /// MaxLevelVertexBot is the max of the max cell values on vertex
         Expected = -1e7;
         for (int I = 0; I < VertexDegree; I++) {
            Expected =
                std::max(Expected, 2 * DefMesh->CellsOnVertexH(IVertex, I));
         }
         Diff = std::abs(DefVertCoord->MaxLevelVertexBotH(IVertex) - Expected);
         if (Diff > 1e-10) {
            Err += 1;
         }
      }

      /// Determine test pass/fail
      if (Err == 0) {
         LOG_INFO("VertCoordTest: minMaxLevelVertex PASS");
      } else {
         LOG_INFO("VertCoordTest: minMaxLevelVertex FAIL");
         RetVal += 1;
      }

      // Finalize Omega objects
      VertCoord::clear();
      HorzMesh::clear();
      Dimension::clear();
      Halo::clear();
      Decomp::clear();
      MachEnv::removeAll();
   }
   Kokkos::finalize();
   MPI_Finalize();

   if (RetVal >= 256)
      RetVal = 255;

   return RetVal;

} // end of main
//===-----------------------------------------------------------------------===/
