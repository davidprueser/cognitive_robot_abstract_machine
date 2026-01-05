import pandas as pd
from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    LearnRSPN,
)
from krrood.entity_query_language.entity import let, an, entity
from matplotlib import pyplot as plt

from probabilistic_model.learning.jpt.jpt import JPT
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.relational.rspns import Person, Government, Supports, Nation, Region, \
    Adjacent, Conflict, nation_part_decomposition, RSPNTemplate, region_part_decomposition, class_spec_nation, \
    class_spec_region
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


    template = RSPNTemplate(class_spec=class_spec_region)
    template.probabilistic_circuit.plot_structure()
    plt.show()
    # for part in template.sub_rspns:
    #     part.probabilistic_circuit.plot_structure()
    #     plt.show()

    grounded = template.ground(region)
    # grounded.probabilistic_circuit.plot_structure()
    # plt.show()

    learned_nation = LearnRSPN(Region, region, class_spec_region)
    # learned_nation.probabilistic_circuit.plot_structure()
    # plt.show()

    # learned_nation = LearnRSPN(Nation, [n1, knowrob_nation], class_spec_nation)
    learned_nation.probabilistic_circuit.plot_structure()
    plt.show()

    grounded_learned_nation = learned_nation.ground(region)
    # grounded_learned_nation.probabilistic_circuit.plot_structure()
    # plt.show()



if __name__ == "__main__":
    example()
