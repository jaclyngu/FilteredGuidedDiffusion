from FGDBasicExperiment import *
from FGDExperimentSet import *

class FGDStructureExperimentSet(FGDExperimentSet):
    ExperimentClass = FGDBasicExperiment

    def __init__(self, path=None, **kwargs):
        super(FGDStructureExperimentSet, self).__init__(path=path, **kwargs);
        self._result_frames = [];



    def getPathForParams(self,
                         guide_image=None,
                         guide_text='',
                         target_dstructure = 0.3,
                         mask_strength=1.0,
                         **kwargs):
        rstring =  "{}_{}_{}_targetStruct{}_maskstr{}".format(
            self.__class__.ExperimentClass.__name__,
            guide_image.file_name_base,
            guide_text,
            target_dstructure,
            mask_strength
        )
        return os.path.join(self.getDir('results'), rstring)+os.sep;