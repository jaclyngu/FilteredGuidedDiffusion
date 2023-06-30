from .FGDExperiment import *
import matplotlib.pyplot as plt
from apy import *
from apy.amedia import *
from apy.aobject.AExperimentDirectory import *
import os

def ImageRow(*images):
    totalw = 0;
    maxh = 0;
    for im in images:
        totalw = totalw+im.width;
        if(im.height>maxh):
            maxh = im.height;
    rim = Image.Zeros([maxh,totalw, 3]);
    currentw = 0;
    for im in images:
        rim.pixels[:im.height,currentw:currentw+im.width,:]=im.fpixels;
        currentw = currentw+im.width;
    return rim;

class FGDExperimentSet(AExperimentDirectory):

    ExperimentClass = FGDExperiment;

    def __init__(self, path=None, **kwargs):
        super(FGDExperimentSet, self).__init__(path=path, **kwargs);

    def getPathForParams(self,**kwargs):
        rstring = self.__class__.ExperimentClass.GetParamString(**kwargs)
        return os.path.join(self.getDir('results'), rstring)+os.sep;

    def loadResultFromPath(self, path=None):
        return self.__class__.ExperimentClass(path=path);

    def evaluateResult(self, **kwargs):
        experiment = self.__class__.ExperimentClass.Create(
            directory=self.getPathForParams(**kwargs),
            **kwargs);
        experiment.saveJSON()
        experiment.Evaluate()

        def getResultPath(fname):
            return os.path.join(experiment.getDir("results"), fname);

        ImageRow(experiment.results[0][1], experiment.getGuideScaledForComparison()).writeToFile(getResultPath("ResultComparison.png"));
        experiment.showdStructureValues();
        plt.savefig(getResultPath("ControlPlot.pdf"));
        experiment.results[0][0].writeToFile(getResultPath("DirectOutputResult.png"))
        experiment.results[0][1].writeToFile(getResultPath("UpsampledOutputResult.png"))
        experiment.saveJSON();
        return experiment

    def ResultForParams(self, **kwargs):
        return super(FGDExperimentSet, self).ResultForParams(**kwargs);

    # <editor-fold desc="Property: 'experimentParams'">
    @property
    def experimentParams(self):
        return self.getInfo("experimentParams");
    @experimentParams.setter
    def experimentParams(self, value):
        self.setInfo('experimentParams', value);
    # </editor-fold>


