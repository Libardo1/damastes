Codebase for performing posterior inference on a generative model of tabular data.


Example interactive usage:

From a Python terminal:
>> import data
>> start_state = data.load_botany() # Load the botany dataset as a State object
>> import mcmc
>> trace = mcmc.mcmc(star_state, n_iters=500) # Run 500 iterations of MCMC
>> trace.save(‘my_analysis’) # Save the set of samples to disk
>> number_of_relations = trace.compute(lambda state: len(unique(state.rels.zr)))) # Compute the number of inferred relations for each sample in the trace
>> plt.hist(number_of_relations) # Plot the posterior over the number of unique relations mentioned in the dataset
>> trace.samples[-1].plot() #Output an HTML file visualizing the last state of the trace for inspection


To run end-to-end inference:
# TODO(malmaud): Fill in