import pandas as pd
from krrood.entity_query_language.entity import let, an, entity
from matplotlib import pyplot as plt

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.relational.rspns import Person, Government, Supports, Nation, Region, \
    Adjacent, Conflict, nation_part_decomposition, RSPNTemplate
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import ProbabilisticCircuit


def example():
    david = Person("David", 25)
    tom = Person("Tom", 27)
    checker_chan = Person("Simon Wallukat", 28)

    cdu = Government(funny=True)
    persons = [david, tom, checker_chan]
    supporters = [Supports(checker_chan, cdu)]
    n1 = Nation(government=cdu, supporters=supporters, persons=persons)

    daniel = Person("Daniel", 50)
    tede = Person("Tede", 25)
    daniel_union = Government(funny=False)

    knowrob_supporters = [daniel, tede]

    knowrob_nation = Nation(
        government=daniel_union,
        persons=[daniel, tede],
        supporters=[Supports(daniel, daniel_union), Supports(tede, daniel_union)],
        gdp=-10,
    )

    region = Region(
        nations=[n1, knowrob_nation],
        adjacency=[Adjacent(n1, knowrob_nation)],
        conflicts=[Conflict(n1, knowrob_nation)],
    )

    n2 = let(Nation, [])
    p3 = let(Person, [])
    q = an(
        entity(
            n2,
            n2.government.funny == True,
            n2.persons == [david, tom, p3],
            )
    )
    template = RSPNTemplate(region)
    rspn = template.ground()
    rspn.probabilistic_circuit.plot_structure()
    plt.show()
    # ground_region = rspn.ground(region)
    # # pretty_print_ground(ground_region)
    # ground_region.probabilistic_circuit.plot_structure()
    # plt.show()
    # print(ground_region.is_decomposable())

    # Try the LearnRSPN algorithm on the same example
    from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
        LearnRSPN,
    )

    # # Learn an RSPN for Region from a single example region
    # region_schema = CLASS_SCHEMA[Region]
    # evidence_database = pandas.DataFrame([region.__dict__])
    # V_region = (
    #         list(region_schema.unique_parts)
    #         + list(region_schema.exchangeable_parts)
    # )
    # learned_region = LearnRSPN(Region, [region], V_region)
    # learned_region.probabilistic_circuit.plot_structure()
    # plt.show()
    # print("Learned Region RSPN decomposable:", learned_region.is_decomposable())

    # Optionally, also learn for Nation using both nations from the region
    # part_decomposition = nation_part_decomposition
    # evidence_database = []
    # # for nation in region.nations:
    #
    #
    # persons = [david, tom, checker_chan]
    # l = [25, 27, 26]
    # name = ["David", "Tom", "Simon"]
    # df_data = {
    #     "age": l,
    #     "name": name
    # }
    # df = pd.DataFrame(df_data)
    #
    # variables = infer_variables_from_dataframe(df)
    # jpt = JPT(variables)
    # jpt = jpt.fit(df)
    # jpt.plot_structure()
    # plt.show()
    #
    # learned_nation = LearnRSPN(part_decomposition, region.nations)
    # # list von nations ist db
    # learned_nation.plot_structure()
    # plt.show()
    # print(
    #     "Learned Nation RSPN decomposable:",
    #     learned_nation.probabilistic_circuit.is_decomposable(),
    # )
    #
    # except Exception as e:
    #     print("LearnRSPN demo failed:", e)


if __name__ == "__main__":  # quick demo if this file is run
    example()

    # # collect all classes that need persistence
    # all_classes = {Nation, Adjacent, Conflict, Supports, Region, Person, Government}
    #
    # class_diagram = ClassDiagram(
    #     list(sorted(all_classes, key=lambda c: c.__name__, reverse=True))
    # )
    #
    # instance = ORMatic(
    #     class_dependency_graph=class_diagram,
    # )
    #
    # instance.make_all_tables()
    #
    # file_path = os.path.join(os.path.dirname(__file__), "dataset", "ormatic_interface.py")
    #
    # with open(file_path, "w") as f:
    #     instance.to_sqlalchemy_file(f)
