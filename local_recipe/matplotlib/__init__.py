from pythonforandroid.recipe import PythonRecipe

class MatplotlibRecipe(PythonRecipe):

    version = '3.8.3'

    url = 'https://github.com/matplotlib/matplotlib/archive/refs/tags/v{version}.zip'

    depends = ['python3', 'numpy', 'setuptools']

    python_depends = ['pillow', 'cycler', 'fonttools', 'kiwisolver', 'packaging', 'pyparsing']

    call_hostpython_via_targetpython = False

    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['MPLCONFIGDIR'] = '/tmp'
        return env

recipe = MatplotlibRecipe()