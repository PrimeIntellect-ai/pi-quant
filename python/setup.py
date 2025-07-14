import os
import subprocess
import multiprocessing

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

CMAKE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')  # Go up one directory to find CMakeLists.txt
)
NUM_JOBS = max(multiprocessing.cpu_count() - 1, 1)  # Use all but one core


class BuildException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class CMakeBuildExtension(Extension):
    def __init__(self, name, root_dir=''):
        super().__init__(name, sources=[])
        self.root_dir = os.path.abspath(root_dir)


class CMakeBuildExecutor(build_ext):
    def initialize_options(self):
        super().initialize_options()

    def run(self):
        try:
            print(subprocess.check_output(['cmake', '--version']))
        except OSError:
            raise BuildException(
                'CMake must be installed to build the piquant binaries from source. Please install CMake and try again.'
            )
        super().run()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        if os.path.exists(self.build_temp):
            import shutil

            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(os.path.join(self.build_lib, 'piquant')),
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        build_args = [
            '--target piquant',
            '-j' + str(NUM_JOBS),
            '-v',
        ]
        print(subprocess.check_call(['cmake', ext.root_dir] + cmake_args, cwd=self.build_temp))
        print(subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp))


setup(
    name='pypiquant',
    author='Mario Sieg',
    author_email='mario@primeintellect.ai',
    packages=['piquant'],
    package_dir={'': 'src'},  # tell setuptools packages are under src/
    package_data={
        'piquant': ['libquant.so', 'libquant.dylib', 'libquant.dll'],
    },
    include_package_data=True,
    ext_modules=[CMakeBuildExtension('piquant', root_dir=CMAKE_ROOT)],
    cmdclass={
        'build_ext': CMakeBuildExecutor,
    },
    zip_safe=False,
)
