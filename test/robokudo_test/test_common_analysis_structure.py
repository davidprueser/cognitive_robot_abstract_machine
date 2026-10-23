from __future__ import annotations
import robokudo.common_analysis_structure
import robokudo.types.annotation
import robokudo.types.scene


class TestCommonAnalysisStructureObject(object):
    cls = robokudo.common_analysis_structure.CommonAnalysisStructure

    def test_annotations(self) -> None:
        common_analysis_structure = self.cls()

        classification = robokudo.types.annotation.Classification()
        classification.source = "classifier"
        color1 = robokudo.types.annotation.SemanticColor()
        color1.color = "black"
        color2 = robokudo.types.annotation.SemanticColor()
        color2.color = "red"

        object_hypothesis = robokudo.types.scene.ObjectHypothesis()
        object_hypothesis.annotations.append(classification)
        object_hypothesis.annotations.append(color1)
        object_hypothesis.annotations.append(color2)

        classification_list = common_analysis_structure.filter_by_type(
            robokudo.types.annotation.Classification, object_hypothesis.annotations
        )
        color_list = common_analysis_structure.filter_by_type(
            robokudo.types.annotation.SemanticColor, object_hypothesis.annotations
        )

        assert len(classification_list) == 1
        assert len(color_list) == 2
        assert color_list[0] == color1
        assert color_list[1] == color2

    def test_ohs_with_criteria(self) -> None:
        common_analysis_structure = self.cls()

        classification = robokudo.types.annotation.Classification()
        classification.source = "classifier"
        classification.classname = "Spoon"
        color1 = robokudo.types.annotation.SemanticColor()
        color1.color = "black"
        color2 = robokudo.types.annotation.SemanticColor()
        color2.color = "red"

        object_hypothesis = robokudo.types.scene.ObjectHypothesis()
        object_hypothesis.annotations.append(classification)
        object_hypothesis.annotations.append(color1)
        object_hypothesis.annotations.append(color2)
        second_object_hypothesis = robokudo.types.scene.ObjectHypothesis()

        filtered_object_hypotheses = (
            common_analysis_structure.filter_by_type_and_criteria(
                robokudo.types.annotation.Classification,
                object_hypothesis.annotations,
                {"classname": ("==", "Spoon")},
            )
        )
        assert len(filtered_object_hypotheses) == 1

        # Attribute checking is case-sensitive!
        filtered_object_hypotheses = (
            common_analysis_structure.filter_by_type_and_criteria(
                robokudo.types.annotation.Classification,
                second_object_hypothesis.annotations,
                {"classname": ("==", "spoon")},
            )
        )
        assert len(filtered_object_hypotheses) == 0

        # Other Object without annotations
        filtered_object_hypotheses = (
            common_analysis_structure.filter_by_type_and_criteria(
                robokudo.types.annotation.Classification,
                second_object_hypothesis.annotations,
                {"classname": ("==", "Spoon")},
            )
        )
        assert len(filtered_object_hypotheses) == 0

        filtered_object_hypotheses = (
            common_analysis_structure.filter_by_type_and_criteria(
                robokudo.types.annotation.SemanticColor,
                object_hypothesis.annotations,
                {"color": ("==", "red")},
            )
        )
        assert len(filtered_object_hypotheses) == 1
