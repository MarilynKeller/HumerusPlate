from Cython.Distutils import build_ext
import platform
import os

# if 'FRANKENGEIST_ROOT_DIR' in os.environ:
#     import sys
#     sys.path.insert(0, os.environ['FRANKENGEIST_ROOT_DIR'])

from setup_helpers import Extension, setup_extended

sourcefiles = ['sample2meshdist.pyx']
additional_options = {'include_dirs': []}

# if 'FRANKENGEIST_EIGEN' in os.environ:
#     additional_options['include_dirs'].append(os.environ['FRANKENGEIST_EIGEN'])
# # otherwise Eigen should be on the system


if platform.system().lower() in ['darwin', 'linux']:
    import sysconfig
    extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
    extra_compile_args += ["-std=c++11"]
    additional_options['extra_compile_args'] = extra_compile_args


setup_extended(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("sample2meshdist", sourcefiles, language="c++", **additional_options)],
    include_dirs=['.'],
)
