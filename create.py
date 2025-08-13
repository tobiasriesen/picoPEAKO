import picopeako

td = picopeako.TrainingData(specfiles_in=['tmp/example.nc'], num_spec=10)
td.mark_random_spectra()
td.save_training_data()
