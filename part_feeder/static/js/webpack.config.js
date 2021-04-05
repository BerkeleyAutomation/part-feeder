const path = require('path');

module.exports = {
    entry: "./anim.js",
    output: {
        path: path.resolve(__dirname, '../../assets'),
        filename: 'bundle.js',
    },
    mode: "development",
    optimization: {
        minimize: false,
    },
    module: {
        rules: [
            {
                test: /\.m?js$/,
                exclude: /(node_modules|bower_components)/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env']
                    }
                }
            }
        ]
    }
};
