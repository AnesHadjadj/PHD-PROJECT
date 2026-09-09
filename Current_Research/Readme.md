# Current research

## Active contribution: DSEG-LIME-guided analysis of KBF

DSEG-LIME was adapted to the MGMT CNN by segmenting the FLAIR slice with SAM/DSEG and transferring the perturbation masks to the corresponding KLF input.

- [x] Reproduce the Capuozzo KBF pipeline
- [x] Adapt DSEG-LIME to the MGMT classifier
- [x] Run a balanced 20-patient pilot across five folds
- [x] Test faithfulness, random controls, stability and perturbation sensitivity
- [x] Compare positive DSEG regions with KBF support
- [x] Generate 585 fold-0 training explanations
- [x] Run the first fold-0 reweighting sentinel
- [ ] Confirm results under the exact historical training protocol
- [ ] Evaluate improved integration mechanisms
- [ ] Perform five-fold validation of the selected contribution

See the [complete DSEG-LIME/KBF research record](./DSEG_LIME_KBF/).
