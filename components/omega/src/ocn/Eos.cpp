#include "Eos.h"
#include "DataTypes.h"
#include "HorzMesh.h"

namespace OMEGA {

Eos *Eos::DefaultEos = nullptr;
std::map<std::string, std::unique_ptr<Eos>> Eos::AllEos;

TEOS10Poly75t::TEOS10Poly75t() {}

LinearEOS::LinearEOS() {}

Eos::Eos(const std::string &Name, ///< [in] Name for eos object
         const HorzMesh *Mesh,    ///< [in] Horizontal mesh
         int NVertLevels          ///< [in] Number of vertical levels
) {
   SpecVol = Array2DReal("SpecVol", Mesh->NCellsSize, NVertLevels);
   SpecVolDisplaced =
       Array2DReal("SpecVolDisplaced", Mesh->NCellsSize, NVertLevels);
   // SpecVol = Array2DReal("SpecVol", Mesh->NCellsSize, NVertLevels);
   // Array dimension lengths
   NCellsAll = Mesh->NCellsAll;
   NChunks   = NVertLevels / VecLength;

} // end constructor

// Initialize the Eos and its options. Assumes Hormesh is initialized
int Eos::init() {

   int Err               = 0;
   HorzMesh *DefHorzMesh = HorzMesh::getDefault();
   I4 NVertLevels        = DefHorzMesh->NVertLevels;

   // Create default eos
   Eos::DefaultEos = create("Default", DefHorzMesh, NVertLevels);

   // Get EosConfig group
   Config *OmegaConfig = Config::getOmegaConfig();
   Config EosConfig("Eos");
   Err = OmegaConfig->get(EosConfig);
   if (Err != 0) {
      LOG_CRITICAL("Eos: Eos group not found in Config");
      return Err;
   }
   std::string EosTypeStr;
   Err = EosConfig.get("EosType", EosTypeStr);
   if (Err != 0) {
      LOG_CRITICAL("Eos: EosType not found in "
                   "EosConfig");
      return Err;
   }

   if (EosTypeStr == "Linear") {
      DefaultEos->eosChoice = EosType::Linear;
   } else if (EosTypeStr == "teos10") {
      DefaultEos->eosChoice = EosType::TEOS10Poly75t;
   } else {
      LOG_CRITICAL("Eos: Unknown EosType requested");
      Err = -1;
      return Err;
   }

   Err = EosConfig.get("LinearDRhoDT", DefaultEos->lineardRhodT);
   if (Err != 0) {
      LOG_CRITICAL("Eos: linear dRhodT not found in "
                   "EosConfig");
      return Err;
   }
   Err = EosConfig.get("LinearDRhoDS", DefaultEos->lineardRhodS);
   if (Err != 0) {
      LOG_CRITICAL("Eos: linear dRhodS not found in "
                   "EosConfig");
      return Err;
   }
   Err = EosConfig.get("LinearRhoT0S0", DefaultEos->linearRhoT0S0);
   if (Err != 0) {
      LOG_CRITICAL("Eos: Ref Rho linearRhoT0S0 not found in "
                   "EosConfig");
      return Err;
   }

   LOG_INFO("set default values {}, {}, {}", DefaultEos->lineardRhodT,
            DefaultEos->lineardRhodS, DefaultEos->linearRhoT0S0);
   return Err;
} // end init

KOKKOS_FUNCTION void
Eos::computeSpecVolLinear(Array2DReal &SpecVol, I4 ICell, I4 KChunk,
                          const Array2DReal &ConservativeTemperature,
                          const Array2DReal &AbsoluteSalinity,
                          const Array2DReal &Pressure) {

   const I4 KStart = KChunk * VecLength;
   for (int KVec = 0; KVec < VecLength; ++KVec) {
      const I4 K = KStart + KVec;
      SpecVol(ICell, K) =
          1.0 /
          (linearRhoT0S0 + (lineardRhodT * ConservativeTemperature(ICell, K) +
                            lineardRhodS * AbsoluteSalinity(ICell, K)));
   }
}

KOKKOS_FUNCTION void
Eos::computeSpecVolTEOS10Poly75t(Array2DReal &SpecVol, I4 ICell, I4 KChunk,
                                 const Array2DReal &ConservativeTemperature,
                                 const Array2DReal &AbsoluteSalinity,
                                 const Array2DReal &Pressure) {

   const I4 KStart = KChunk * VecLength;
   for (int KVec = 0; KVec < VecLength; ++KVec) {
      const I4 K        = KStart + KVec;
      SpecVol(ICell, K) = 1.0;
   }
}

void Eos::computeSpecVol(Array2DReal &SpecVol,
                         const Array2DReal &ConservativeTemperature,
                         const Array2DReal &AbsoluteSalinity,
                         const Array2DReal &Pressure) {
   OMEGA_SCOPE(LocSpecVol, SpecVol);
   OMEGA_SCOPE(LocComputeSpecVolLinear, computeSpecVolLinear);
   OMEGA_SCOPE(LocComputeSpecVolTEOS10Poly75t, computeSpecVolTEOS10Poly75t);

   if (eosChoice == EosType::Linear) {
      parallelFor(
          "eos-linear", {NCellsAll, NChunks},
          KOKKOS_LAMBDA(I4 ICell, I4 KChunk) {
             LocComputeSpecVolLinear(LocSpecVol, ICell, KChunk,
                                     ConservativeTemperature, AbsoluteSalinity,
                                     Pressure);
          });
   } else if (eosChoice == EosType::TEOS10Poly75t) {
      parallelFor(
          "eos-teos10", {NCellsAll, NChunks},
          KOKKOS_LAMBDA(I4 ICell, I4 KChunk) {
             LocComputeSpecVolTEOS10Poly75t(LocSpecVol, ICell, KChunk,
                                            ConservativeTemperature,
                                            AbsoluteSalinity, Pressure);
          });
   }
}

// void Eos::computeSpecVolLinear(Array2DReal &SpecVol, I4 ICell, I4 KChunk,
//                                const Array2DReal &ConservativeTemperature,
//                                const Array2DReal &AbsoluteSalinity,
//                                const Array2DReal &Pressure) {
//
//    const I4 KStart = KChunk * VecLength;
//    for (int KVec = 0; KVec < VecLength; ++KVec) {
//       const I4 K = KStart + KVec;
//       SpecVol(ICell, K) =
//           1.0 /
//           (linearRhoT0S0 + (lineardRhodT * ConservativeTemperature(ICell, K)
//           +
//                             lineardRhodS * AbsoluteSalinity(ICell, K)));
//    }
// }
//
// void Eos::computeSpecVolTEOS10Poly75t(
//     Array2DReal &SpecVol, I4 ICell, I4 KChunk,
//     const Array2DReal &ConservativeTemperature,
//     const Array2DReal &AbsoluteSalinity, const Array2DReal &Pressure) {
//
//    const I4 KStart = KChunk * VecLength;
//    for (int KVec = 0; KVec < VecLength; ++KVec) {
//       const I4 K        = KStart + KVec;
//       SpecVol(ICell, K) = 1.0;
//    }
// }

//------------------------------------------------------------------------------
// Destroys the eos class
Eos::~Eos() {

   // No operations needed, Kokkos arrays removed when no longer in scope

} // end destructor

//------------------------------------------------------------------------------
// Removes all eos instances before exit
void Eos::clear() { AllEos.clear(); } // end clear

//------------------------------------------------------------------------------
// Removes eos from list by name
void Eos::erase(const std::string &Name) { AllEos.erase(Name); } // end erase

//------------------------------------------------------------------------------
// Get default eos
Eos *Eos::getDefault() { return Eos::DefaultEos; } // end get default

//------------------------------------------------------------------------------
// Get eos by name
Eos *Eos::get(const std::string &Name ///< [in] Name of eos
) {

   auto it = AllEos.find(Name);

   if (it != AllEos.end()) {
      return it->second.get();
   } else {
      LOG_ERROR("Eos::get: Attempt to retrieve non-existent eos:");
      LOG_ERROR("{} has not been defined or has been removed", Name);
      return nullptr;
   }

} // end get eos

} // namespace OMEGA
