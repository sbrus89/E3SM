//===-- SplitExplicitInit.cpp - split-explicit initialization --*- C++ -*-===//
//
// Utilities for initializing split-explicit time stepping state.
//
//===----------------------------------------------------------------------===//

#include "SplitExplicitInit.h"
#include "Config.h"
#include "Error.h"
#include "GlobalConstants.h"
#include "Logging.h"
#include "OmegaKokkos.h"

#include <algorithm>
#include <cmath>

namespace OMEGA {

//------------------------------------------------------------------------------
SplitExplicitConfig
SplitExplicitInit::readConfigOptions(const TimeInterval &TimeStep) {

   SplitExplicitConfig SEConfig;
   SEConfig.BtrTimeStep = TimeStep;

   Config *OmegaConfig = Config::getOmegaConfig();
   Config TimeIntConfig("TimeIntegration");
   Error Err = OmegaConfig->get(TimeIntConfig);
   CHECK_ERROR_ABORT(Err, "TimeIntegration group not found in Config");

   bool IsUnsplit = false;
   std::string TimeStepperStr;
   if (TimeIntConfig.get("TimeStepper", TimeStepperStr).isSuccess()) {
      if (TimeStepperStr.rfind("Unsplit", 0) == 0) {
         IsUnsplit            = true;
         SEConfig.SplitFactor = 0._Real;
      }
   }

   if (!IsUnsplit) {
      std::string BtrTimeStepStr;
      if (TimeIntConfig.get("BtrTimeStep", BtrTimeStepStr).isSuccess()) {
         SEConfig.BtrTimeStep = TimeInterval(BtrTimeStepStr);
      }
   }

   I4 NTimeStepIteration = 1;
   if (TimeIntConfig.get("NTimeStepIteration", NTimeStepIteration)
           .isSuccess()) {
      if (NTimeStepIteration < 1) {
         ABORT_ERROR("NTimeStepIteration must be greater than zero");
      }
      SEConfig.NTimeStepIteration = NTimeStepIteration;
   }

   I4 NBclCoriolisIteration = 2;
   if (TimeIntConfig.get("NBclCoriolisIteration", NBclCoriolisIteration)
           .isSuccess()) {
      if (NBclCoriolisIteration < 1) {
         ABORT_ERROR("NBclCoriolisIteration must be greater than zero");
      }
      SEConfig.NBclCoriolisIteration = NBclCoriolisIteration;
   }
   if (IsUnsplit) {
      SEConfig.NBclCoriolisIteration = 1;
   }

   if (!IsUnsplit) {
      SEConfig.NBtrSubcycles =
          computeSubcycleCount(TimeStep, SEConfig.BtrTimeStep);
   }

   return SEConfig;
}

//------------------------------------------------------------------------------
SplitExplicitBarotropicStepperType
SplitExplicitInit::getBtrTimeStepperFromStr(const std::string &InString) {

   if (InString == "Predictor-Corrector") {
      return SplitExplicitBarotropicStepperType::PredictorCorrector;
   }

   ABORT_ERROR("BtrTimeStepper should be 'Predictor-Corrector' but got {}",
               InString);
   return SplitExplicitBarotropicStepperType::Invalid;
}

//------------------------------------------------------------------------------
I4 SplitExplicitInit::computeSubcycleCount(const TimeInterval &TimeStep,
                                           const TimeInterval &BtrTimeStep) {

   R8 TimeStepSeconds;
   R8 BtrTimeStepSeconds;
   TimeStep.get(TimeStepSeconds, TimeUnits::Seconds);
   BtrTimeStep.get(BtrTimeStepSeconds, TimeUnits::Seconds);

   if (BtrTimeStepSeconds <= 0.) {
      ABORT_ERROR("BtrTimeStep must be greater than zero");
   }

   return std::max<I4>(
       1, static_cast<I4>(std::ceil(TimeStepSeconds / BtrTimeStepSeconds)));
}

//------------------------------------------------------------------------------
void SplitExplicitInit::allocateScratch(SplitExplicitScratch &Scratch,
                                        const HorzMesh *Mesh,
                                        const VertCoord *VCoord,
                                        const std::string &Name) {

   if (!Mesh)
      LOG_CRITICAL("Invalid mesh");
   if (!VCoord)
      LOG_CRITICAL("Invalid vertical coordinate");

   Scratch.NormalBarotropicVelocitySubcycleCur = Array1DReal(
       "NormalBarotropicVelocitySubcycleCur" + Name, Mesh->NEdgesSize);
   Scratch.NormalBarotropicVelocitySubcycleNew = Array1DReal(
       "NormalBarotropicVelocitySubcycleNew" + Name, Mesh->NEdgesSize);
   Scratch.BarotropicPressureAnomalySubcycleCur = Array1DReal(
       "BarotropicPressureAnomalySubcycleCur" + Name, Mesh->NCellsSize);
   Scratch.BarotropicPressureAnomalySubcycleNew = Array1DReal(
       "BarotropicPressureAnomalySubcycleNew" + Name, Mesh->NCellsSize);
   Scratch.BarotropicPressure =
       Array1DReal("BarotropicPressure" + Name, Mesh->NCellsSize);
   Scratch.BarotropicForcing =
       Array1DReal("BarotropicForcing" + Name, Mesh->NEdgesSize);
   Scratch.BarotropicFlux =
       Array1DReal("BarotropicFlux" + Name, Mesh->NEdgesSize);
   Scratch.BaseVelocityTend = Array2DReal(
       "BaseVelocityTend" + Name, Mesh->NEdgesSize, VCoord->NVertLayers);

   deepCopy(Scratch.NormalBarotropicVelocitySubcycleCur, 0.);
   deepCopy(Scratch.NormalBarotropicVelocitySubcycleNew, 0.);
   deepCopy(Scratch.BarotropicPressureAnomalySubcycleCur, 0.);
   deepCopy(Scratch.BarotropicPressureAnomalySubcycleNew, 0.);
   deepCopy(Scratch.BarotropicPressure, 0.);
   deepCopy(Scratch.BarotropicForcing, 0.);
   deepCopy(Scratch.BarotropicFlux, 0.);
   deepCopy(Scratch.BaseVelocityTend, 0.);
}

} // namespace OMEGA
