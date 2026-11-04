#pragma once

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
  RandomizedQRColumnPivoting  // NYI
};


