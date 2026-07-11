from pathlib import Path
import unittest

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from feature_extraction import extract_feature_maps
from region_proposal import propose_regions


class NavigationStateTests(unittest.TestCase):
    def test_review_label_and_page_survive_reruns_and_navigation(self):
        image=np.full((180,300,3),155,np.uint8)
        cv2.rectangle(image,(80,55),(220,130),(70,130,195),-1)
        features=extract_feature_maps(image)
        result=propose_regions(image,features,"navigation_test",min_area=20,max_regions=3)
        self.assertTrue(result.proposals)
        image_path=Path("/private/tmp/navigation_test.png")
        cv2.imwrite(str(image_path),image)

        app=AppTest.from_file("app.py",default_timeout=20).run()
        state={
            "image_path":image_path,
            "image_name":"navigation_test.png",
            "processed":image,
            "feature_maps":features,
            "proposal_result":result,
        }
        for key,value in state.items():
            app.session_state[key]=value

        navigation=next(widget for widget in app.radio if widget.key=="active_page")
        navigation.set_value("Human Review / Labeling").run()
        self.assertEqual(app.session_state["active_page"],"Human Review / Labeling")

        label=next(widget for widget in app.selectbox if widget.key=="label_R001")
        label.set_value("weld_irregularity_candidate").run()
        self.assertEqual(app.session_state["active_page"],"Human Review / Labeling")
        self.assertEqual(app.session_state["label_R001"],"weld_irregularity_candidate")

        save=next(widget for widget in app.button if widget.label=="Save Review Metadata")
        save.click().run()
        self.assertEqual(app.session_state["active_page"],"Human Review / Labeling")
        self.assertEqual(app.session_state["label_R001"],"weld_irregularity_candidate")

        next(widget for widget in app.radio if widget.key=="active_page").set_value("Overview").run()
        self.assertEqual(app.session_state["active_page"],"Overview")
        next(widget for widget in app.radio if widget.key=="active_page").set_value("Human Review / Labeling").run()
        restored=next(widget for widget in app.selectbox if widget.key=="label_R001")
        self.assertEqual(restored.value,"weld_irregularity_candidate")
        self.assertFalse(app.exception)


if __name__=="__main__": unittest.main()
