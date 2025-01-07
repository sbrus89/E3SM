#ifndef OMEGA_EOS_H
#define OMEGA_EOS_H
//===-- ocn/Eos.h - Equation of State --------------------*- C++ -*-===//
//
/// \file
/// \brief Contains functors for calculating specific volume
///
/// This header defines functors to be called by the time-stepping scheme
/// to calculate the specific volume based on the choice of EOS
//
//===----------------------------------------------------------------------===//

#include "AuxiliaryState.h"
#include "Config.h"
#include "HorzMesh.h"
#include "MachEnv.h"
#include "OceanState.h"
#include "TimeMgr.h"
#include "Tracers.h"
#include "EosConstants.h"

#include <cmath>

namespace OMEGA {

/// TEOS10 75-term polynomial
class TEOS10Poly75t {
 public:
   // bool Enabled;

   /// constructor declaration
   TEOS10Poly75t(); 

   /// The functor takes point-wise conservative temperature, absolute salinity
   // and pressure as inputs and outputs the specific volume
   KOKKOS_FUNCTION Real operator() (Real Sa, Real Ct, Real P) const {
      Real SpecVol = refprofile(P) + calcdelta(Sa, Ct, P);
      return SpecVol;
   }
   KOKKOS_FUNCTION Real calcdelta(Real Sa, Real Ct, Real P) const {
      const Real SAu = 40 * 35.16504/35;
      const Real CTu = 40.;
      const Real Pu = 1e4;
      const Real DeltaS = 24.;
      GSW_SPECVOL_COEFFICIENTS;
      const Real ss = sqrt ( (Sa+DeltaS)/SAu );
      Real tt = Ct/CTu;
      Real pp = P/Pu;
      Real vp5 = V005;

      Real vp4 = V014 * tt + V104 * ss + V004; 
      Real vp3 = ( V023 * tt + V113 * ss + V013 ) * tt  
	      + ( V203 * ss + V103 ) * ss + V003;
      Real vp2 = ( ( ( V042 * tt + V132 * ss + V032 ) * tt  
                 + ( V222 * ss + V122 ) * ss + V022 ) * tt  
		 + ( ( V312 * ss + V212 ) * ss + V112 ) * ss + V012 ) * tt  
	         + ( ( ( V402 * ss + V302 ) * ss + V202 ) * ss + V102 ) * ss + V002;
      Real vp1 = ( ( ( ( V051 * tt + V141 * ss + V041 ) * tt  + ( V231 * ss + V131 ) * ss + V031 ) * tt  
                 + ( ( V321 * ss + V221 ) * ss + V121 ) * ss + V021 ) * tt  
		 + ( ( ( V411 * ss + V311 ) * ss + V211 ) * ss + V111 ) * ss + V011 ) * tt  
	         + ( ( ( ( V501 * ss + V401 ) * ss + V301 ) * ss + V201 ) * ss + V101 ) * ss + V001;
      Real vp0 = ( ( ( ( ( V060 * tt + V150 * ss + V050 ) * tt  + ( V240 * ss + V140 ) * ss + V040 ) * tt
                 + ( ( V330 * ss + V230 ) * ss + V130 ) * ss + V030 ) * tt  
                 + ( ( ( V420 * ss + V320 ) * ss + V220 ) * ss + V120 ) * ss + V020 ) * tt  
		 + ( ( ( ( V510 * ss + V410 ) * ss + V310 ) * ss + V210 ) * ss + V110 ) * ss + V010 ) * tt 
	      + ((((( V600 * ss + V500 ) * ss + V400 ) * ss + V300 ) * ss + V200 ) * ss + V100 ) * ss + V000;
    
      Real delta = ( ( ( ( vp5 * pp + vp4 ) * pp + vp3 ) * pp + vp2 ) * pp + vp1 ) * pp + vp0;
      return delta;
   }
   KOKKOS_FUNCTION Real refprofile(Real P) const {
      const Real Pu = 1e4;
      const Real V00 = -4.4015007269e-05;
      const Real V01 = 6.9232335784e-06; 
      const Real V02 = -7.5004675975e-07;
      const Real V03 = 1.7009109288e-08;
      const Real V04 = -1.6884162004e-08;
      const Real V05 = 1.9613503930e-09;
      Real pp = P/Pu;

      Real v0 = (((((V05 * pp+V04) * pp+V03 ) * pp+V02 ) * pp+V01) * pp+V00) * pp;
      return v0;
   } 


};

/// Linear Equation of State
// class LinearEOS {
//  public:
//    bool Enabled;
// 
//    /// constructor declaration
//    LinearEOS();
// 
//    /// The functor takes edge index, vertical chunk index, and arrays for
//    /// normalized relative vorticity, normalized planetary vorticity, layer
//    /// thickness on edges, and normal velocity on edges as inputs,
//    /// outputs the tendency array
//    KOKKOS_FUNCTION Real operator()(const Array2DReal &Tend, I4 IEdge, I4 KChunk,
//                                    const Array2DReal &NormRVortEdge,
//                                    const Array2DReal &NormFEdge,
//                                    const Array2DReal &FluxLayerThickEdge,
//                                    const Array2DReal &NormVelEdge) const {
// 
//       const I4 KStart         = KChunk * VecLength;
//       Real VortTmp[VecLength] = {0};
// 
//       for (int J = 0; J < NEdgesOnEdge(IEdge); ++J) {
//          I4 JEdge = EdgesOnEdge(IEdge, J);
//          for (int KVec = 0; KVec < VecLength; ++KVec) {
//             const I4 K    = KStart + KVec;
//             Real NormVort = (NormRVortEdge(IEdge, K) + NormFEdge(IEdge, K) +
//                              NormRVortEdge(JEdge, K) + NormFEdge(JEdge, K)) *
//                             0.5_Real;
// 
//             VortTmp[KVec] += WeightsOnEdge(IEdge, J) *
//                              FluxLayerThickEdge(JEdge, K) *
//                              NormVelEdge(JEdge, K) * NormVort;
//          }
//       }
// 
//       for (int KVec = 0; KVec < VecLength; ++KVec) {
//          const I4 K = KStart + KVec;
//          Tend(IEdge, K) += VortTmp[KVec];
//       }
//    }
// 
//  private:
//    Array1DI4 NEdgesOnEdge;
//    Array2DI4 EdgesOnEdge;
//    Array2DReal WeightsOnEdge;
// };

} // namespace OMEGA
#endif
