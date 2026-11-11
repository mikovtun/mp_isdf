#pragma once

// Select where Kokkos evaluation runs. Auto uses the accelerator when this
// build enables CUDA, otherwise the host. Device requires a CUDA build.
enum class EvaluationSpace { Auto, Host, Device };

// Defines function on an axis
enum class AxisType {
  AtomicOrbital,
  AtomicOrbitalGradX,
  AtomicOrbitalGradY,
  AtomicOrbitalGradZ,
  OccupiedOrbital
};

// Defines algorithm that drives AnchorSelection class, distributed. These are tiled in SLATE.
enum class DistributedAnchorSelectionAlgorithm {
  SVD,
  RandomizedQRColumnPivoting  // NYI
};

// Defines algorithm that drives AnchorSelection class, local. These are non-tiled and are single-node
enum class LocalAnchorSelectionAlgorithm {
  QRColumnPivoting,
  OnlineQRColumnPivoting,
  RandomizedFeatureKhatriRao,
  RandomizedAOQuadratureKhatriRao,
  CentroidalVoronoi
};


