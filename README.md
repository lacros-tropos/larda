

## pyLARDA v3
pyLARDA for accessing and analysing ground based remote sensing data. 
It tries to simplify following tasks:

- finding netcdf files in a complex hierarchy
- loading data from differently formatted netcdfs
- stiching data from consecutive files together
- simplify common plotting tasks

Documentation is available at [larda-doc](https://martin-rdz.github.io/larda-doc/)


### Setup
Firstly, clone the repository and make sure all dependencies are installed.
Depending on your datasource of choice:

#### remote
You just need to know the link to the backend backend of choice and may move to Quickstart.

#### local
For local data it is necessary to include the source in a certain directory structure.

```
├── larda        # github managed source code
│   ├── docs
│   ├── examples
│   ├── ListCollector.py
│   ├── pyLARDA  # actual python module
│   ├── README.md
│   ├── requirements.txt
│   └── run_docs.sh
├── larda-cfg  # configuration files (a)
│   ├── campaigns.toml
│   ├── [single campaign].toml
│   └── [single campaign].toml
├── larda-connectordump
│   └── [auto generated subfolder for each campaign]
└── larda-doc           # folder if you want to generate the docs
    └── ... 
```

### Quickstart
Make sure that the module is available at your pythonpath when in doubt use `sys.path.append('dir')`.

```
import pyLARDA

link_to_backend = 'http://...' 
# or use pyLARDA.LARDA('local')
larda = pyLARDA.LARDA('remote', uri=link_to_backend)
print('available campaigns', larda.campaign_list)
larda.connect('campaign_name')
larda.print_params()
MIRA_Zg = larda.read("MIRA","Zg", [dt_begin, dt_end], [0, 4000])
fig, ax = pyLARDA.Transformations.plot_timeheight(MIRA_Zg, range_interval=[500, 3000],
                                                  z_converter='lin2z')
fig.savefig('MIRA_Z.png', dpi=250)
```

For more examples refer to the scripts in the `examples` directory.


### Architecture

![overview on the structure](examples/api.png)

### History
This version of the LACROS research data analyser (LARDA) is based on two prior versions in C and python2 respectively.
Major changes are the migration to python3, netcdf4 and the inclusion of radar Doppler spectra.


### License
Copyright 2019, Johannes Bühl, Martin Radenz, Willi Schimmel, Teresa Vogl
[MIT License](http://www.opensource.org/licenses/mit-license.php)

