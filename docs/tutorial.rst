:github_url: https://github.com/BartKeulen/smartstart

#########
Tutorial
#########
This page contains tutorials on how to use the SmartStart package. The tutorial starts with running Q-Learning on a GridWorld environment and visualizing the results. After that the example is extended to using SmartStart. The last two parts of the tutorial cover running multiple experiments in parallel and saving the data before plotting all the results.

What modules to import is discussed in the parts of the tutorial where they are first presented. It is good practice to group all the import statements at the top of your file in the following order:

    1.  Standard libraries (e.g. time, random)
    2.  3d party libraries (e.g. numpy)
    3.  Own package (in this case smartstart)

Full code examples of the tutorial can be found in the :ref:`examples` section.

==========
Q-Learning
==========
Q-Learning is a temporal difference method that works with tabular environments, i.e. discrete states and actions. For more information on Q-Learning I recommend reading 'Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto`.

Before starting I'll first explain the general structure for running an algorithm and visualizing the results.

    1.  The first step is to construct the environment
    2.  Initialize the learning agent (e.g. Q-Learning)
    3.  Run the training
    4.  Save/Visualize the results

Lets start by constructing the :class:`~smartstart.environments.gridworld.GridWorld` environment and the :class:`~smartstart.environments.gridworldvisualizer.GridWorldvisualizer`. First we have to import the necessary modules. The following code snippets imports the modules and creates the environment and visualizer

.. code-block:: python

   from smartstart.environments.gridworld import GridWorld
   from smartstart.environments.gridworldvisualizer import GridWorldVisualizer

   # Create environment and visualizer
   grid_world = GridWorld.generate(GridWorld.EASY)
   visualizer = GridWorldVisualizer(grid_world)
   visualizer.add_visualizer(GridWorldVisualizer.LIVE_AGENT,
                             GridWorldVisualizer.CONSOLE,
                             GridWorldVisualizer.VALUE_FUNCTION,
                             GridWorldVisualizer.DENSITY)

The first two lines import the necessary modules. After that the GridWorld is constructed first. The visualizer is constructed separately and has to be given a reference to the environment, this gives the visualizer access to information and state about the environment. The visualizer will automatically give a reference of it self to the environment. Next visualizers have to be add to the visualizer. You can choose from four visualizers:

    1.  :attr:`~smartstart.environments.environment.Visualizer.LIVE_AGENT`, renders the location of the agent
    2.  :attr:`~smartstart.environments.environment.Visualizer.CONSOLE`, prints results in a console
    3.  :attr:`~smartstart.environments.environment.Visualizer.VALUE_FUNCTION`, renders the value function as a heath map
    4.  :attr:`~smartstart.environments.environment.Visualizer.DENSITY`, renders the density as a heath map

Now we can go to initializing the agent and running the training. In this example only a few parameters of the agent are used, look at the class or read the documentation to see what parameters are available. Add the following code to your python file (remember to put import statement at the top of your file)

.. code-block:: python

    from smartstart.algorithms.qlearning import QLearning

    # Initialize agent, see class for available parameters
    agent = QLearning(grid_world,
                      alpha=0.1,
                      epsilon=0.05,
                      num_episodes=1000,
                      max_steps=1000,
                      exploration=QLearning.E_GREEDY)

    # Train the agent, summary contains training data
    summary = agent.train(render=True,
                          render_episode=True,
                          print_results=True)

The :class:`~smartstart.algorithms.qlearning.QLearning` agent has one parameter that must be set, which is the environment. All the other parameters have default values but can be set to whatever value you want. The available exploration strategies can be found in the class, there it is also explained how to add more. After initializing the agent the training can be run by calling the :meth:`~smartstart.algorithms.tdlearning.TDLearning.train` method on the agent. Three parameters can be supplied to the train method:

  * render, if ``True`` every time-step will be rendered
  * render_episode, if ``True`` only once after every episode will be rendered
  * print_results, if ``True`` the results will be printed to the terminal. Often set to ``False`` when :attr:`~smartstart.environments.environment.Visualizer.CONSOLE` is active.

The :meth:`~smartstart.algorithms.tdlearning.TDLearning.train` method returns a :class:`~smartstart.utilities.datacontainers.Summary` object containing the results. THe results can be plotted easily using the :meth:`~smartstart.utilities.plot.plot_summary` method. Two parameters have to be provided. The first one is a list of files or a list of :class:`~smartstart.utilities.datacontainers.Summary` objects, a single file or :class:`~smartstart.utilities.datacontainers.Summary` object can also be provided. See :meth:`~smartstart.utilities.plot.plot_summary` method documentation for The second parameter is the plot type, three plot types are provided

  * :meth:`~smartstart.utilities.plot.mean_reward_episode` plots the average reward per episode
  * :meth:`~smartstart.utilities.plot.mean_reward_std_episode` plots the average reward per episode and the standard deviation
  * :meth:`~smartstart.utilities.plot.steps_episode` plots the number of steps per episode

The plotting types also work for a single file or summary object. Now we are going to plot the :class:`~smartstart.utilities.datacontainers.Summary` object directly, later in the tutorial we will plot from files. We are going to plot the average reward per episode and the number of steps per episode.

.. code-block:: python

    from smartstart.utilities.plot import plot_summary, show_plot, \
    mean_reward_episode, steps_episode

    # Plot results
    plot_summary(summary, mean_reward_episode, ma_window=5,
                 title="Easy GridWorld Q-Learning Average Reward per Episode")
    plot_summary(summary, steps_episode, ma_window=5,
                 title="Easy GridWorld Q-Learning Steps per Episode")
    show_plot()

First of it's important to call the :meth:`~smartstart.utilities.plot.show_plot()` after initializing all your plots, this call will actually render the figures and keep them open. In this tutorial we only use a few of the parameters that can be provided to the :meth:`~smartstart.utilities.plot.plot_summary` method. The first two are the summary and the plot type. The ``ma_window=5`` parameter applies a moving average filter to the data with window size 5. The title parameter adds a title to the plot. In later tutorials we will also cover how to automatically save the plots.

Now you should be able to run the full algorithm and visualize the results. The training can be very slow when every time-step is rendered, you can either supply ``render=False`` to the :meth:`~smartstart.algorithms.tdlearning.TDLearning.train` method or close the visualization window, when ``render_episode=True`` it will still be rendered once per episode, otherwise it closes and continues without rendering.

A full code example can be found here: :ref:`qlearning_example`. 

==========
SmartStart
==========
Now we are going to run the same experiment but with SmartStart turned on. There is not much difference, so you can either make a copy of the file you have made for the Q-Learning tutorial or make changes in that file. For running an algorithm with SmartStart we only have to change the agent

.. code-block:: python
  
    from smartstart.smartexploration.smartexploration import generate_smartstart_object

    # Initialize agent, see class for available parameters
    agent = generate_smartstart_object(QLearning,
                                       env=grid_world,
                                       alpha=0.1,
                                       epsilon=0.05,
                                       num_episodes=10,
                                       max_steps=1000,
                                       exploration=QLearning.E_GREEDY)

Instead of defining a separate SmartStart class for every learning algorithm we generate that class using the :meth:`~smartstart.smartexploration.smartexploration.generate_smartstart_object` method. This method takes in the base algorithm to be used (e.g. Q-Learning, SARSA), some specific parameters to SmartStart (see note below) and the usual parameters for the base algorithm. The :meth:`~smartstart.smartexploration.smartexploration.generate_smartstart_object` method will then construct a SmartStart class that inherits the base class. The object returned will work in the same way as the base algorithm but with the SmartStart extension.

A full code example can be found here: :ref:`smartstart_example`. 

.. note::

  The SmartStart class does not show up in documentation, I haven't been able to figure out how to fix this. See the source code for the documentation on the SmartStart class. The SmartStart class is defined within the :meth:`~smartstart.smartexploration.smartexploration.generate_smartstart_object` method.

============
Experimenter
============
Now we are able to run single experiments and visualize the training and results. But often you don't want to run one experiment but multiple experiments. The :meth:`~smartstart.utilities.experimenter.run_experiment` method makes it easy to this. The idea is you define a task function that takes one dictionary as argument, this dictionary contains the specific parameters for that experiment. You then define a dictionary called the parameter grid containing the task and all the different parameters to be used. The :meth:`~smartstart.utilities.experimenter.run_experiment` method will take this parameter grid and turn it into individual sets of parameters and run the task function with each set of parameters. Lets first see what happens to the parameter grid:

.. code-block:: python

  # Original parameter grid
  param_grid = {
    'task': task,
    'num_exp': 2,
    'use_smart_start': [True, False],
  }

  # Will turn into 4 sets of parameters
  params = [
    { 'task': task, 'use_smart_start': True, 'run': 0 },
    { 'task': task, 'use_smart_start': True, 'run': 1 },
    { 'task': task, 'use_smart_start': False, 'run': 2 },
    { 'task': task, 'use_smart_start': False, 'run': 3 }
  ]

As you can see the original parameter grid turns into 4 separate dictionaries. The :meth:`~smartstart.utilities.experimenter.run_experiment` method will run 4 experiments with each experiment using one of the dictionaries from the params list. Now we can start with our full example, open a new python file and in which we are going to define the experiment. 

Since we are running multiple experiments in parallel we have to save the data, so we start by defining a directory to save the data.

.. code-block:: python

    from smartstart.utilities.utilities import get_data_directory

    # Get the path to the data folder in the same directory as this file.
    # If the folder does not exists it will be created
    summary_dir = get_data_directory(__file__)

The ``__file__`` tag is the reference to the current file. The :meth:`smartstart.utilities.utilities.get_data_directory` method takes this reference to the file and creates a directory with the name data in the same folder. The path to the data folder is then returned.

Now we define our task function. Since we have covered most of the parts on how to run a training session we fill in the whole function immediately.

.. code-block:: python

    import random

    import numpy as np

    from smartstart.algorithms.qlearning import QLearning
    from smartstart.smartexploration.smartexploration import generate_smartstart_object
    from smartstart.environments.gridworld import GridWorld

    # Define the task function for the experiment
    def task(params):
        # Reset the seed for random number generation
        random.seed()
        np.random.seed()

        # Create environment
        env = GridWorld.generate(GridWorld.MEDIUM)

        # Here we use a dict to define the parameters, this makes it easy to
        # make sure the experiments use the same parameters
        kwargs = {
            'alpha': 0.1,
            'epsilon': 0.05,
            'num_episodes': 1000,
            'max_steps': 2500,
            'exploration': QLearning.E_GREEDY
        }

        # Initialize agent, check params if it needs to use SmartStart or not
        if params['use_smart_start']:
            agent = generate_smartstart_object(QLearning, env, **kwargs)
        else:
            agent = QLearning(env, **kwargs)

        # Train the agent, summary contains training data. Make sure to set the
        # rendering and printing to False when multiple experiments run in
        # parallel. Else it will consume a lot of computation power.
        summary = agent.train(render=False,
                              render_episode=False,
                              print_results=False)

        # Save the summary. The post_fix parameter can be used to create a unique
        #  file name.
        summary.save(directory=summary_dir, post_fix=params['run'])

Some important things to note here are:
  
  * We set the seed of the :mod:`random` and :mod:`np.random` modules at the begin of our task. The reason for this is the way how multiprocessing works in python. When a function is ran in parallel, each process needs access to the already defined values and imported modules. Python makes a snapshot of the current state and uses that snapshot for each process. :mod:`random` and :mod:`np.random` have already been imported when the snapshot is taken, so is the state of their random number generator! So each process will generate the exact same numbers when the seed is not reset.
  * We use an if loop to choose between using SmartStart and not. Since we want the rest of the parameters to be equal it is better to use a dictionary to supply the keyword arguments then filling them in two times.
  * The summary is saved in the directory. The default name of a summary is algorithm plus environment name and with SmartStart prefixed with SmartStart. In this example the summaries will be named `QLearning_GridWorldMedium` and `SmartStart_QLearning_GridWorldMedium`. With the post_fix argument a unique filename can be created. In this case we only add the ``params['run']`` parameter, which is useful when each experiment is performed multiple times. Since all the parameters are the same for that case ``params['run]`` is added to give it a unique identifier. When you are using different alpha's for example you would add ``alpha=params['alpha']`` to the post-fix for example.

We can now define our parameter grid and run the experiment. The :meth:`~smartstart.utilities.experimenter.run_experiment` method takes next to the parameter grid the number of processes to run in parallel, when this value is set to ``-1`` the number of processes will be equal to the number of cpu-cores. This means all the cores will be used for the experiment.

.. code-block:: python

    from smartstart.utilities.experimenter import run_experiment

    param_grid = {
        'task': task,
        'num_exp': 5,
        'use_smart_start': [True, False]
    }

    run_experiment(param_grid, n_processes=-1)


A full code example can be found here: :ref:`experimenter_example`. 

================
Plotting Results
================
After we have run our experiment we want to visualize the results. This can again be done with our :meth:`~smartstart.utilities.plot.plot_summary` method. We have discussed all the methods used for plotting results, this time there are a few differences. The code below shows how to plot the results obtained with the previous experiment

.. code-block:: python

    import os

    from smartstart.utilities.plot import plot_summary, \
        mean_reward_std_episode, steps_episode, show_plot
    from smartstart.utilities.utilities import get_data_directory

    # Get directory where the summaries are saved. Since it is the same folder as
    #  the experimenter we can use the get_data_directory method
    summary_dir = get_data_directory(__file__)

    # Define the files list
    files = [os.path.join(summary_dir, "QLearning_GridWorldMedium"),
             os.path.join(summary_dir, "SmartStart_QLearning_GridWorldMedium")]

    legend = ["Q-Learning", "SmartStart Q-Learning"]

    # We are going to save the plots in img folder
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'img')

    # Plot average reward and standard deviation per episode
    # When an output directory is supplied the plots will not be rendered with
    # a title. The title is used as filename for the plot.
    plot_summary(files,
                 mean_reward_std_episode,
                 ma_window=5,
                 title="Q-Learning GridWorldMedium Average Reward per Episode",
                 legend=legend,
                 output_dir=output_dir)

    plot_summary(files,
                 steps_episode,
                 ma_window=5,
                 title="Q-Learning GridWorldMedium Steps per Episode",
                 legend=legend,
                 output_dir=output_dir)

    show_plot()

Things to note here:

  * The filenames in the file directory are defined without the ``_*.json`` where the asterix is any of the numbers. You have to supply the filenames like this. The :meth:`~smartstart.utilities.plot.plot_summary` method will search for all files that start with the provided filepath and anything after that until the .json. All the files found with that pattern will be averaged and plotted as one line.
  * We have added a legend to the plots
  * An output directory for the plots is defined, when the output_dir argument is used in the :meth:`~smartstart.utilities.plot.plot_summary` method the figures will be automatically saved in that directory.
  * This time we use the :meth:`~smartstart.utilities.plot.average_reward_std_episode` type instead of :meth:`~smartstart.utilities.plot.average_reward_episode`. Since we have multiple results per experiment this is a good way to show the variation in the results.
  * The format for the step per episode plot is .png instead of the standard and preferred format .eps. The reason for this is the that .eps can't save the transparent area around the mean and will become opaque. You can try it out to see for yourself.