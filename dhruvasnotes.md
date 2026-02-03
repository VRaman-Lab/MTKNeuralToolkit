1. add juliasyntax and formatter in your startup.jl file (along with revise) if you want them. not in scripts.


Build channel logic:





### Improving to new MTK version

https://juliahub.com/blog/what-s-new-with-modelingtoolkit

can put an input as as parameter (see above)


### questions

Why do we have nonlinearsolve in plot_LIF?
Where is this explicit channel being used? Doesn't seem used in HH or Liu

### Code cleaning for Ella/Elouan

Write a short summary of the different functions. eg i'm looking at src/Electrical/utils.jl :

There are several build_channel and build_channel_explicit methods. In what contexts are they used? What types of channel need which?
Something that would help is to add some 'templates'. What do we require a channel to have, programmatically? A channel.p and a channel.n? 
