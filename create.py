import glob

import picopeako

spec_files = glob.glob('tmp/*.nc')

for spec_file in spec_files:
    td = picopeako.TrainingData(specfiles_in=[spec_file], num_spec=5)
    td.mark_random_spectra()
    td.save_training_data()
