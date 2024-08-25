package com.ai.pdfassistant;

import dev.langchain4j.chain.ConversationalRetrievalChain;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.cassandra.AstraDbEmbeddingConfiguration;
import dev.langchain4j.store.embedding.cassandra.AstraDbEmbeddingStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


@Configuration
public class PdfAssistantConfig {

    //embed the given text document to vectors
    // EX:- cat - [0.2,0.9,-0.75]
    @Bean
    public EmbeddingModel embeddingModel() {
        return new AllMiniLmL6V2EmbeddingModel();
    }


    // add the document to embedded data store (astra db)
    @Bean
    public AstraDbEmbeddingStore astraDbEmbeddingStore(){
        String astraToken="";
        String databaseId="";

        return new AstraDbEmbeddingStore(AstraDbEmbeddingConfiguration
                .builder()
                .token(astraToken)
                .databaseId(databaseId)
                .databaseRegion("us-east1")
                .keyspace("default_keyspace")
                .table("pdfchat")
                .dimension(384)
                .build());
    }

    //split the given document into small chunks
    @Bean
    public EmbeddingStoreIngestor embeddingStoreIngestor(){
        return EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(300,0))
                .embeddingModel(embeddingModel())
                .embeddingStore(astraDbEmbeddingStore())
                .build();
    }

    //read embeddings from embedding data store and call the chat LLM
    @Bean
    public ConversationalRetrievalChain conversationalRetrievalChain(){
        return ConversationalRetrievalChain.builder()
                .chatLanguageModel(OpenAiChatModel.withApiKey(""))
                .retriever(EmbeddingStoreRetriever.from(astraDbEmbeddingStore(),embeddingModel()))
                .build();
    }
}
