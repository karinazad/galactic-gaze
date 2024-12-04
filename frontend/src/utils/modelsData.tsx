const modelsData = [
    {
        type: 'VAE',
        description: 'Variational Autoencoders (VAEs) are generative models that learn to encode data into a latent space and then decode it back into the original space. They are particularly useful for generating new data points similar to the training set.',
        hyperparameters: { learning_rate: 0.001, batch_size: 64 },
    },
    {
        type: 'Diffusion',
        description: 'Diffusion models generate data by gradually transforming noise into samples through a series of denoising steps. They have shown remarkable results in generating high-quality images.',
        hyperparameters: { learning_rate: 0.0001, batch_size: 16 },
    },
    {
        type: 'WJS',
        description: 'Walk Jump Sampling (WJS) is a sampling technique that combines random walks with jumps to efficiently explore complex distributions. This method simplifies training and sampling by requiring only a single noise level, making it effective for generative modeling tasks.',
        hyperparameters: { learning_rate: 0.0005, batch_size: 128, jump_probability: 0.1 },
    },
    {
        type: 'Normalizing Flows',
        description: 'Normalizing Flows are a class of generative models that transform a simple distribution into a complex one using a series of invertible transformations. This allows for exact likelihood computation and efficient sampling from complex distributions.',
        hyperparameters: { learning_rate: 0.001, batch_size: 32, num_layers: 5 },
    },
];

export default modelsData;