from picopeako import Peako

pk = Peako()
best_params, best_quality = pk.train('tmp/example.nc', {
    't_avg': [0, 1, 2],
    'h_avg': [0, 1, 2],
    'span': [0.1, 0.2, 0.3],
    'width': [0.1, 0.2, 0.3],
    'prom': [0.1, 0.2, 0.3],
    'polyorder': [0, 1, 2]
})

print(f"Best parameters: {best_params} with quality {best_quality}")
