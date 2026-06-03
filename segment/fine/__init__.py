"""Fine-grained stage: 3D-TransUNet wrapped as a custom nnU-Net trainer with a
confidence-constrained loss.

Importing this subpackage is enough for nnU-Net to discover the trainers because
they subclass ``nnUNetTrainer`` and live on the import path; nnU-Net's
``recursive_find_python_class`` also searches this package when configured.
"""
