from setuptools import setup, find_packages

# separate git-based dependencies
requirements = []
dependency_links = []

with open('requirements.txt') as f:
    for line in f.read().splitlines():
        if line.startswith('git+'):
            link = line.replace('git+', '')
            dependency_links.append(link)
            requirements.append(link.split('/')[-1].split('.')[0])
        else:
            requirements.append(line)

setup(
    name='avdc-flowdiffusion',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    dependency_links=dependency_links,
    author='AVDC Author',
    author_email='author@example.com',
    description='AVDC Flow Diffusion Model.',
    url='https://github.com/path/to/AVDC',
    python_requires='>=3.6',
)
