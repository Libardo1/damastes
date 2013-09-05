Codebase for performing posterior inference on a generative model of tabular data.
See https://docs.google.com/a/google.com/document/d/1MJTTWiMSWzVuLph7-6usaYD0otW-GRIaqT5QBYwuAak/edit for details.

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


To run end-to-end batch mode with Blaze:
TODO(malmaud): Currently not possible since the Pandas version on Google3 needs to be updated to at least verion .11.

Run blaze on the analyze_tables target, passing in the flags defined in analyze_tables.py.
The batch processer expects to be a given a list of table IDs.
It loads the content of those tables from Webtables, preprocesses it, and saves the processed tables as a State pickle object. It then performs inference.
It then performs inference on a chain starting from that state for the given number of iterations, storing the results as a pickled Trace object in a user-specified location.
 In subsequent executions, analyze_tables will try to load the pickled initial state object from a cache instead of re-loading the tables from Webtables, unless a flag is given to force a reload.


